.import pandas as pd
import numpy as np
import sys
import os

# Transcript-Based OPLM & Pullback Strategy Log
sys.stdout = open('V20_MARKETGUIDE_OPLM_LOG.txt', 'w')
print("🚀 V20 MarketGuide OPLM (False Breakdown + Pullback & 10% Strict SL)...\n", flush=True)

CFG = dict(
    lots=1,
    entry_start=pd.Timestamp('09:17').time(),
    entry_end=pd.Timestamp('15:20').time(), # 3:19 PM scalping window included
    square_off=pd.Timestamp('15:25').time(), # Strictly NO BTST
    lookback=6,             
    reclaim_bars=2, 
    max_trades_per_day=2,   # 2 Trades per day (as per setup availability)
    max_sl_pct=0.10,        # Strict 10% Stop Loss (Transcript Rule: Max 10% SL / 5-8 points)
    target_1_pct=0.50,      # 50% Target 1
    target_2_pct=1.00,      # 100% Target 2 (2x Move)
    jackpot_pct=3.00,       # 300% Target 3 (3x Move)
    sl_buffer=0.02,         # Trailing buffer
)

INSTR = {
    'NIFTY':  dict(step=50, lot=75, scale=1.0, allowed_dte=[0, 1, 2], min_premium=15, max_premium=60), # Value Price (Saste Options)
    'SENSEX': dict(step=100, lot=10, scale=3.3, allowed_dte=[0, 1], min_premium=20, max_premium=80),
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

def calculate_indicators(df):
    df['ema_9'] = df['close'].ewm(span=9, adjust=False).mean()
    df['ema_21'] = df['close'].ewm(span=21, adjust=False).mean()
    return df

def resample_chart(df, timeframe='1min'):
    df = df.set_index('timestamp')
    ohlc = df['close'].resample(timeframe).ohlc().dropna().reset_index()
    ohlc = calculate_indicators(ohlc)
    return ohlc

def detect_false_breakdown(df, i, lookback, reclaim_bars, side_type='CE'):
    """
    OPLM (Option Premium Liquidity Mapping) - Detects False Breakdown / Breakout
    Transcript Rule: Wait for low to break, trap retail, then reclaim (Pullback & Continuation).
    """
    if i < lookback + reclaim_bars: return False
    o, h, l, c = df['open'].values, df['high'].values, df['low'].values, df['close'].values
    
    if side_type == 'CE':
        # False Breakdown (Support broken, then reclaimed)
        lvl_lo = l[i - lookback - reclaim_bars + 1 : i - reclaim_bars + 1].min()
        return l[i-1] < lvl_lo and c[i] > lvl_lo and c[i] > o[i]
    else:
        # False Breakout (Resistance broken, then rejected)
        lvl_hi = h[i - lookback - reclaim_bars + 1 : i - reclaim_bars + 1].max()
        return h[i-1] > lvl_hi and c[i] < lvl_hi and c[i] < o[i]

def run_marketguide_backtest(index_type, months):
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
            
            # Using 1-min timeframe for precise scalping & 3:19 PM momentum
            spot_1m = resample_chart(day_spot, '1min')
            if len(spot_1m) < 20: continue
            
            trades_today = 0
            
            for i in range(max(CFG['lookback'] + CFG['reclaim_bars'], 10), len(spot_1m)):
                ts = spot_1m.iloc[i]['timestamp']
                if ts.time() < CFG['entry_start'] or ts.time() > CFG['entry_end']: continue
                if trades_today >= CFG['max_trades_per_day']: break
                
                spot_close = spot_1m.iloc[i]['close']
                spot_ema = spot_1m.iloc[i]['ema_21']
                atm = round(spot_close / meta['step']) * meta['step']
                
                for side in ['CE', 'PE']:
                    if trades_today >= CFG['max_trades_per_day']: break
                    
                    # Trend Filter (Avoid fighting the immediate momentum)
                    if side == 'CE' and spot_close < spot_ema: continue
                    if side == 'PE' and spot_close > spot_ema: continue
                    
                    # OTM Selection for Value Price (Saste Options)
                    target_strike = atm + (meta['step'] * 2) if side == 'CE' else atm - (meta['step'] * 2)
                    opt_series = day_opt[(day_opt['strike'] == target_strike) & (day_opt['option_type'] == side)].copy()
                    
                    if opt_series.empty: continue
                    
                    opt_series = opt_series.sort_values('timestamp').reset_index(drop=True)
                    opt_1m = opt_series.set_index('timestamp')['close'].resample('1min').ohlc().dropna().reset_index()
                    opt_i_list = opt_1m[opt_1m['timestamp'] == ts].index
                    if len(opt_i_list) == 0: continue
                    opt_i = opt_i_list[0]
                    
                    # Detect False Breakdown on Option Chart (Value Price Entry)
                    opt_setup = detect_false_breakdown(opt_1m, opt_i, CFG['lookback'], CFG['reclaim_bars'], side_type='CE')
                    
                    if opt_setup or (ts.time() >= pd.Timestamp('15:15').time() and ts.time() <= pd.Timestamp('15:19').time()):
                        trade_reason = "3:19_SCALP" if ts.time() >= pd.Timestamp('15:15').time() else "OPLM_FALSE_BREAKDOWN"
                        
                        entry_win = opt_series[(opt_series['timestamp'] >= ts) & (opt_series['timestamp'] <= ts + pd.Timedelta(minutes=3))]
                        if entry_win.empty: continue
                        
                        entry_px = entry_win.iloc[0]['close']
                        entry_ts = entry_win.iloc[0]['timestamp']
                        
                        if not (meta['min_premium'] <= entry_px <= meta['max_premium']): continue
                        
                        # Strict 10% Stop Loss from Entry (Transcript Rule)
                        sl_px = entry_px * (1 - CFG['max_sl_pct'])
                        
                        exit_px = None
                        reason = None
                        future = opt_series[opt_series['timestamp'] > entry_ts]
                        
                        for _, f in future.iterrows():
                            # Trailing SL on Targets
                            if f['high'] >= (entry_px * (1 + CFG['target_1_pct'])):
                                sl_px = entry_px * 1.10 # Trail to 10% profit
                            if f['high'] >= (entry_px * (1 + CFG['target_2_pct'])):
                                sl_px = entry_px * 1.50 # Trail to 50% profit
                                
                            # Exit Conditions
                            if f['low'] <= sl_px:
                                exit_px = sl_px
                                reason = 'STRICT_SL_HIT' if sl_px < entry_px else 'TRAILING_SL'
                                break
                            if f['high'] >= (entry_px * (1 + CFG['jackpot_pct'])): 
                                exit_px = entry_px * (1 + CFG['jackpot_pct'])
                                reason = '3X_JACKPOT'
                                break
                            if f['timestamp'].time() >= CFG['square_off']:
                                exit_px = f['close']
                                reason = 'NO_BTST_SQUAREOFF'
                                break
                        
                        if exit_px is None:
                            exit_px = future.iloc[-1]['close'] if not future.empty else entry_px
                            reason = 'TIME_EXIT'
                        
                        pnl = (exit_px - entry_px) * meta['lot'] - 40 # Assuming 40 rs brokerage
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
print("\n" + "="*50 + "\nNIFTY OPLM BACKTEST\n" + "="*50, flush=True)
nifty_df = run_marketguide_backtest('NIFTY', ['01', '02', '03', '04', '05'])

print("\n" + "="*50 + "\nSENSEX OPLM BACKTEST\n" + "="*50, flush=True)
sensex_df = run_marketguide_backtest('SENSEX', ['01', '02', '03', '04', '05'])

# ============ SUMMARY ============
df = pd.concat([nifty_df, sensex_df], ignore_index=True)

if df.empty:
    print("❌ 0 Trades. Wait for False Breakdown Setup.")
else:
    print("\n" + "="*55, flush=True)
    print("🏆 FINAL RESULT (MarketGuide OPLM Strategy)", flush=True)
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
    print(f"COMBINED: {len(df)} trades | PnL: ₹{total_pnl:,.2f}", flush=True)
    print(f"{'='*55}", flush=True)
    
    df.to_csv('V20_MARKETGUIDE_OPLM_RESULTS.csv', index=False)
    print("\n✅ Saved: V20_MARKETGUIDE_OPLM_RESULTS.csv", flush=True)
