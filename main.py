import pandas as pd
import numpy as np
import sys
import gc
import os

sys.stdout = open('V19_SUPER_HYBRID_LOG.txt', 'w')
print("🚀 V19 Super Hybrid (EMA + Dual Sweep + 1-Lot Jackpot)...\n", flush=True)

CFG = dict(
    lots=1,
    entry_start=pd.Timestamp('09:17').time(),
    entry_end=pd.Timestamp('14:45').time(),
    square_off=pd.Timestamp('15:15').time(),
    lookback=6,             # AI Optimized (30 min)
    reclaim_bars=2, 
    min_break_pts=3, 
    max_trades_per_day=1,   # 1 Trade per day
    be_trigger=1.30,        # 30% profit pe breakeven
    sl_buffer=0.02,         # 2% trailing buffer
)

INSTR = {
    'NIFTY':  dict(step=50, lot=65, scale=1.0, allowed_dte=[0, 1, 4], min_premium=10, max_premium=70),
    'SENSEX': dict(step=100, lot=20, scale=3.3, allowed_dte=[0, 1], min_premium=20, max_premium=150),
}

def load_month(month, index_type='NIFTY'):
    if index_type == 'NIFTY':
        opt_file = f'data/monthly/nifty_2026-{month}.parquet'
        spot_file = f'data/monthly/nifty_spot_2026-{month}.parquet'
    else:
        opt_file = f'data/monthly_sensex/sensex_2026-{month}.parquet'
        spot_file = f'data/monthly_sensex/sensex_spot_2026-{month}.parquet'
        
    if not os.path.exists(opt_file) or not os.path.exists(spot_file): return None, None
    try:
        opt = pd.read_parquet(opt_file)
        spot = pd.read_parquet(spot_file)
    except: return None, None
    
    for df in [opt, spot]:
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        if df['timestamp'].dt.tz is not None: df['timestamp'] = df['timestamp'].dt.tz_localize(None)
    opt['expiry'] = pd.to_datetime(opt['expiry'])
    if opt['expiry'].dt.tz is not None: opt['expiry'] = opt['expiry'].dt.tz_localize(None)
    opt['date'] = opt['timestamp'].dt.date
    opt['dte'] = (opt['expiry'].dt.normalize() - opt['timestamp'].dt.normalize()).dt.days
    spot['date'] = spot['timestamp'].dt.date
    return opt, spot

def calculate_indicators(df_5m):
    df_5m['ema_50'] = df_5m['close'].ewm(span=50, adjust=False).mean()
    return df_5m

def resample_5m(df):
    df = df.set_index('timestamp')
    ohlc = df['close'].resample('5min').ohlc().dropna().reset_index()
    ohlc = calculate_indicators(ohlc)
    return ohlc

def detect_sweep_on_chart(df_5m, i, lookback, reclaim_bars, side_type='CE'):
    if i < lookback + reclaim_bars: return False
    o, h, l, c = df_5m['open'].values, df_5m['high'].values, df_5m['low'].values, df_5m['close'].values
    
    if side_type == 'CE':
        lvl_lo = l[i - lookback - reclaim_bars + 1 : i - reclaim_bars + 1].min()
        return l[i] <= lvl_lo and c[i] > lvl_lo and c[i] > o[i]
    else:
        lvl_hi = h[i - lookback - reclaim_bars + 1 : i - reclaim_bars + 1].max()
        return h[i] >= lvl_hi and c[i] < lvl_hi and c[i] < o[i]

def run_super_hybrid_backtest(index_type, months):
    meta = INSTR[index_type]
    all_trades = []
    
    for month in months:
        opt, spot = load_month(month, index_type)
        if opt is None: continue
        dates = sorted(opt['date'].unique())
        
        for today in dates:
            day_opt = opt[opt['date'] == today].copy()
            day_spot = spot[spot['date'] == today].copy()
            if len(day_opt) < 50 or len(day_spot) < 30: continue
            
            dte_today = int(day_opt['dte'].min())
            if dte_today not in meta['allowed_dte']: continue
            
            current_expiry = day_opt[day_opt['dte'] == dte_today]['expiry'].iloc[0]
            day_opt = day_opt[day_opt['expiry'] == current_expiry].copy()
            
            spot_5m = resample_5m(day_spot)
            if len(spot_5m) < 20: continue
            
            trades_today = 0
            
            for i in range(max(CFG['lookback'] + CFG['reclaim_bars'], 10), len(spot_5m)):
                ts = spot_5m.iloc[i]['timestamp']
                if ts.time() < CFG['entry_start'] or ts.time() > CFG['entry_end']: continue
                if trades_today >= CFG['max_trades_per_day']: break
                
                spot_close = spot_5m.iloc[i]['close']
                spot_ema = spot_5m.iloc[i]['ema_50']
                atm = round(spot_close / meta['step']) * meta['step']
                
                for side in ['CE', 'PE']:
                    if trades_today >= CFG['max_trades_per_day']: break
                    
                    # Trend Filter
                    if side == 'CE' and spot_close < spot_ema: continue
                    if side == 'PE' and spot_close > spot_ema: continue
                    
                    target_strike = atm + meta['step'] if side == 'CE' else atm - meta['step']
                    opt_series = day_opt[(day_opt['strike'] == target_strike) & (day_opt['option_type'] == side)].copy()
                    
                    if opt_series.empty:
                        opt_series = day_opt[(day_opt['strike'] == atm) & (day_opt['option_type'] == side)].copy()
                        target_strike = atm
                    if opt_series.empty: continue
                    
                    opt_series = opt_series.sort_values('timestamp').reset_index(drop=True)
                    opt_5m = opt_series.set_index('timestamp')['close'].resample('5min').ohlc().dropna().reset_index()
                    opt_i_list = opt_5m[opt_5m['timestamp'] == ts].index
                    if len(opt_i_list) == 0: continue
                    opt_i = opt_i_list[0]
                    
                    # Dual Chart Detection
                    spot_sweep = detect_sweep_on_chart(spot_5m, i, CFG['lookback'], CFG['reclaim_bars'], side_type=side)
                    opt_sweep = detect_sweep_on_chart(opt_5m, opt_i, CFG['lookback'], CFG['reclaim_bars'], side_type='CE')
                    
                    if spot_sweep or opt_sweep:
                        trade_reason = "DUAL_SWEEP" if (spot_sweep and opt_sweep) else ("SPOT_SWEEP" if spot_sweep else "OPTION_SWEEP")
                        
                        entry_win = opt_series[(opt_series['timestamp'] > ts) & (opt_series['timestamp'] <= ts + pd.Timedelta(minutes=10))]
                        if entry_win.empty: continue
                        
                        entry_px, entry_ts, red_low = None, None, None
                        for _, fr in entry_win.iterrows():
                            if not (meta['min_premium'] <= fr['close'] <= meta['max_premium']): continue
                            if fr['close'] < fr['open'] and red_low is None: red_low = fr['low']
                            if fr['close'] > fr['open'] and red_low is not None:
                                entry_px, entry_ts = fr['close'], fr['timestamp']
                                break
                        
                        if entry_px is None or red_low >= entry_px or (entry_px - red_low) > (entry_px * 0.40): continue
                        
                        # Hybrid Jackpot Exit
                        sl_px = red_low
                        exit_px = None
                        reason = None
                        breakeven_triggered = False
                        future = opt_series[opt_series['timestamp'] > entry_ts]
                        
                        for _, f in future.iterrows():
                            # 30% Profit pe Breakeven
                            if not breakeven_triggered and f['high'] >= (entry_px * CFG['be_trigger']):
                                sl_px = entry_px
                                breakeven_triggered = True
                            
                            # 2% Trailing
                            if breakeven_triggered:
                                potential_sl = f['low'] - (entry_px * CFG['sl_buffer'])
                                if potential_sl > sl_px: sl_px = potential_sl
                            
                            if f['low'] <= sl_px:
                                exit_px = sl_px
                                reason = 'BE_HIT' if sl_px == entry_px else 'TRAILING_SL'
                                break
                            if f['high'] >= (entry_px * 5.0): # 5x Jackpot
                                exit_px = entry_px * 5.0
                                reason = '5X_JACKPOT'
                                break
                            if f['timestamp'].time() >= CFG['square_off']:
                                exit_px = f['close']
                                reason = 'SQUAREOFF'
                                break
                        
                        if exit_px is None:
                            exit_px = future.iloc[-1]['close'] if not future.empty else entry_px
                            reason = 'TIME_EXIT'
                        
                        pnl = (exit_px - entry_px) * meta['lot'] - 40
                        trades_today += 1
                        
                        all_trades.append({
                            'Date': today, 'Index': index_type, 'Signal': side, 'Strike': target_strike,
                            'Entry': round(entry_px, 2), 'Exit': round(exit_px, 2), 
                            'PnL': round(pnl, 2), 'Result': reason, 'Trigger': trade_reason
                        })
                        print(f"  {today} {entry_ts.time()} | {side} | {target_strike} | {entry_px:.2f}→{exit_px:.2f} [{reason}] | PnL: ₹{pnl:.0f}", flush=True)
                        break
                        
    return pd.DataFrame(all_trades)

# ============ MAIN RUN ============
print("\n" + "="*50 + "\nNIFTY BACKTEST\n" + "="*50, flush=True)
nifty_df = run_super_hybrid_backtest('NIFTY', ['01', '02', '03', '04', '05'])

print("\n" + "="*50 + "\nSENSEX BACKTEST\n" + "="*50, flush=True)
sensex_df = run_super_hybrid_backtest('SENSEX', ['01', '02', '03', '04', '05'])

# ============ SUMMARY ============
df = pd.concat([nifty_df, sensex_df], ignore_index=True)

if df.empty:
    print("❌ 0 Trades")
else:
    print("\n" + "="*55, flush=True)
    print("🏆 FINAL RESULT (Super Hybrid)", flush=True)
    print("="*55, flush=True)
    
    for idx in ['NIFTY', 'SENSEX']:
        sub = df[df['Index'] == idx]
        if len(sub) == 0: continue
        wins = len(sub[sub['PnL'] > 0])
        print(f"\n--- {idx} ---", flush=True)
        print(f"  Trades: {len(sub)} | WR: {wins/len(sub)*100:.1f}% | PnL: ₹{sub['PnL'].sum():,.2f}", flush=True)
        print(f"  Reason Breakdown:", flush=True)
        print(sub['Result'].value_counts().to_string(), flush=True)
    
    total_pnl = df['PnL'].sum()
    print(f"\n{'='*55}", flush=True)
    print(f"COMBINED: {len(df)} trades | PnL: ₹{total_pnl:,.2f} | Return: {total_pnl/100000*100:.2f}%", flush=True)
    print(f"{'='*55}", flush=True)
    
    df.to_csv('V19_SUPER_HYBRID_RESULTS.csv', index=False)
    print("\n✅ Saved: V19_SUPER_HYBRID_RESULTS.csv", flush=True)
