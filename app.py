#!/usr/bin/env python3
"""DailyPulse — Markets, World News & AI Assistant"""

import subprocess, sys, os, json, threading, re, math
from pathlib import Path
from datetime import datetime

for _pkg in ['flask','yfinance','feedparser','requests','numpy','pandas']:
    try: __import__(_pkg)
    except ImportError:
        print(f'Installing {_pkg}...')
        subprocess.check_call([sys.executable,'-m','pip','install',_pkg,'-q'],
                              stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

from flask import Flask, render_template_string, jsonify, request
import yfinance as yf
import feedparser
import requests
import numpy as np

_UA = {'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36'}

app = Flask(__name__)
CFG_PATH = Path(__file__).parent / 'config.json'

DEFAULT_CFG = {
    'watchlist': ['AAPL','MSFT','NVDA','GOOGL','TSLA'],
    'anthropic_key': '', 'port': 5055,
    'portfolio': [],   # [{symbol, shares, cost}]
    'alerts': [],      # [{symbol, target, condition, note}]
}

def cfg():
    if CFG_PATH.exists():
        try: return {**DEFAULT_CFG, **json.loads(CFG_PATH.read_text())}
        except: pass
    return DEFAULT_CFG.copy()

def save_cfg(data):
    c = cfg(); c.update(data); CFG_PATH.write_text(json.dumps(c, indent=2))

# ── constants ──────────────────────────────────────────────────────────────────
INDICES = [('S&P 500','^GSPC'),('Dow Jones','^DJI'),('Nasdaq','^IXIC'),
           ('Russell 2K','^RUT'),('VIX','^VIX')]

SECTORS = [('Technology','XLK'),('Financials','XLF'),('Health Care','XLV'),
           ('Energy','XLE'),('Cons. Disc.','XLY'),('Cons. Staples','XLP'),
           ('Industrials','XLI'),('Materials','XLB'),('Real Estate','XLRE'),
           ('Utilities','XLU'),('Communication','XLC')]

ASSETS = [('Gold','GLD'),('Silver','SLV'),('Crude Oil','USO'),('Nat. Gas','UNG'),
          ('Bitcoin','BTC-USD'),('Ethereum','ETH-USD')]

SCREENER = ['AAPL','MSFT','NVDA','GOOGL','AMZN','META','TSLA','AVGO','ORCL','NFLX',
            'JPM','BAC','GS','BRK-B','V','MA','WFC','C',
            'JNJ','LLY','UNH','PFE','ABBV','MRK',
            'XOM','CVX','COP','SLB',
            'HD','WMT','COST','MCD','NKE','SBUX',
            'CAT','DE','BA','RTX','HON',
            'AMD','INTC','QCOM','CRM','ADBE','NOW','PLTR','SNOW','COIN']

NEWS_FEEDS = {
    'Global': [
        ('BBC World',      'http://feeds.bbci.co.uk/news/world/rss.xml'),
        ('Reuters World',  'https://feeds.reuters.com/reuters/worldNews'),
        ('Al Jazeera',     'https://www.aljazeera.com/xml/rss/all.xml'),
        ('NPR World',      'https://feeds.npr.org/1004/rss.xml'),
    ],
    'Geopolitics': [
        ('Foreign Policy', 'https://foreignpolicy.com/feed/'),
        ('The Diplomat',   'https://thediplomat.com/feed/'),
        ('CFR',            'https://www.cfr.org/rss/all'),
        ('Belfer Center',  'https://www.belfercenter.org/rss.xml'),
    ],
    'Policy': [
        ('Politico',       'https://rss.politico.com/politics-news.xml'),
        ('The Hill',       'https://thehill.com/rss/syndicator/19109'),
        ('NPR Politics',   'https://feeds.npr.org/1014/rss.xml'),
        ('Axios',          'https://api.axios.com/feed/'),
    ],
    'Nat. Security': [
        ('Defense One',     'https://www.defenseone.com/rss/all/'),
        ('War on Rocks',    'https://warontherocks.com/feed/'),
        ('Breaking Defense','https://breakingdefense.com/feed/'),
        ('Lawfare',         'https://www.lawfaremedia.org/feed'),
    ],
    'Economy': [
        ('CNBC Economy',  'https://www.cnbc.com/id/10000664/device/rss/rss.html'),
        ('MarketWatch',   'https://feeds.content.dowjones.io/public/rss/mw_topstories'),
        ('Investopedia',  'https://www.investopedia.com/feedbuilder/feed/getfeed/?feedName=rss_headline'),
    ],
}

# ── technical analysis ─────────────────────────────────────────────────────────
def calc_rsi(closes, period=14):
    try:
        d = np.diff(closes[-period*2:])
        gains = np.where(d>0,d,0); losses = np.where(d<0,-d,0)
        ag = np.mean(gains[-period:]); al = np.mean(losses[-period:])
        return round(100-(100/(1+ag/al)),1) if al else 100.0
    except: return None

def calc_macd(closes):
    try:
        import pandas as pd
        s = pd.Series(closes)
        m = s.ewm(span=12,adjust=False).mean() - s.ewm(span=26,adjust=False).mean()
        sig = m.ewm(span=9,adjust=False).mean()
        h = m - sig
        cross = ('bullish' if h.iloc[-1]>0 and h.iloc[-2]<=0 else
                 'bearish' if h.iloc[-1]<0 and h.iloc[-2]>=0 else
                 'above' if h.iloc[-1]>0 else 'below')
        return {'macd':round(m.iloc[-1],4),'signal':round(sig.iloc[-1],4),
                'hist':round(h.iloc[-1],4),'cross':cross}
    except: return None

_spy_cache = {'r': None, 'ts': 0}
def spy_returns():
    now = datetime.now().timestamp()
    if _spy_cache['r'] is not None and now-_spy_cache['ts'] < 600:
        return _spy_cache['r']
    try:
        h = yf.Ticker('^GSPC').history(period='1y',interval='1d')
        r = h['Close'].pct_change().dropna().values
        _spy_cache['r'] = r; _spy_cache['ts'] = now; return r
    except: return None

def signal_score(rsi, macd_cross, pct52, mom5):
    s = 0
    if rsi:
        s += 2 if rsi<30 else 1 if rsi<45 else -2 if rsi>70 else -1 if rsi>55 else 0
    if macd_cross == 'bullish': s += 2
    elif macd_cross == 'bearish': s -= 2
    elif macd_cross == 'above': s += 1
    elif macd_cross == 'below': s -= 1
    if mom5: s += 1 if mom5>5 else -1 if mom5<-5 else 0
    if pct52: s += 1 if pct52<10 else -1 if pct52>90 else 0
    s = max(-3,min(3,s))
    labels = ['Strong Sell','Sell','Weak Sell','Neutral','Weak Buy','Buy','Strong Buy']
    colors = ['#f85149','#f85149','#d29922','#8b949e','#3fb950','#3fb950','#00d084']
    return {'score':s,'label':labels[s+3],'color':colors[s+3]}

# ── data fetchers ──────────────────────────────────────────────────────────────
def get_quote(symbol):
    try:
        h = yf.Ticker(symbol).history(period='5d',interval='1d')
        if len(h)<2: return None
        p=h['Close'].iloc[-1]; prev=h['Close'].iloc[-2]
        chg=p-prev; pct=chg/prev*100
        return {'symbol':symbol,'price':round(float(p),4),'change':round(float(chg),4),
                'pct':round(float(pct),2),'volume':int(h['Volume'].iloc[-1])}
    except: return None

def get_analysis(symbol):
    try:
        h = yf.Ticker(symbol).history(period='1y',interval='1d')
        if len(h)<30: return None
        c=h['Close'].values; v=h['Volume'].values
        price=c[-1]; prev=c[-2]; pct=(price-prev)/prev*100
        hi52=h['High'].max(); lo52=h['Low'].min()
        pct52=(price-lo52)/(hi52-lo52)*100 if hi52!=lo52 else 50
        rets=np.diff(c)/c[:-1]
        vol30=round(np.std(rets[-30:],ddof=1)*math.sqrt(252)*100,1)
        rsi=calc_rsi(c); macd=calc_macd(c)
        avg20v=np.mean(v[-20:]); vol_r=v[-1]/avg20v if avg20v else 1
        mom5=(c[-1]-c[-5])/c[-5]*100 if len(c)>=5 else 0
        mom20=(c[-1]-c[-20])/c[-20]*100 if len(c)>=20 else 0
        spr=spy_returns()
        beta=None
        if spr is not None and len(rets)>10:
            n=min(len(rets),len(spr)); sr=spr[-n:]; tr=rets[-n:]
            try:
                cov=np.cov(tr,sr)[0][1]; var=np.var(sr,ddof=1)
                beta=round(cov/var,2) if var else None
            except: pass
        sma20=np.mean(c[-20:])
        sma50=np.mean(c[-50:]) if len(c)>=50 else None
        sma200=np.mean(c[-200:]) if len(c)>=200 else None
        mc=macd['cross'] if macd else None
        sig=signal_score(rsi,mc,pct52,mom5)
        flag=bool(abs(pct)>2.5 or vol_r>2 or abs(sig['score'])>=2)
        return {
            'symbol':symbol,'price':round(float(price),2),'pct':round(float(pct),2),
            'hi52':round(float(hi52),2),'lo52':round(float(lo52),2),'pct52':round(float(pct52),1),
            'rsi':rsi,'macd':macd,'beta':beta,'vol30':vol30,
            'vol_r':round(float(vol_r),2),'mom5':round(float(mom5),2),'mom20':round(float(mom20),2),
            'sma20':round(float(sma20),2),
            'sma50':round(float(sma50),2) if sma50 is not None else None,
            'sma200':round(float(sma200),2) if sma200 is not None else None,
            'signal':sig,'volume':int(v[-1]),'avg20v':int(avg20v),'flag':flag,
        }
    except: return None

_news_cache = {'data': {}, 'ts': 0}
_digest_cache = {'data': None, 'ts': 0}
def get_news(category='all'):
    now = datetime.now().timestamp()
    if now-_news_cache['ts']<600 and _news_cache['data']:
        d=_news_cache['data']
        return [a for arts in d.values() for a in arts] if category=='all' else d.get(category,[])

    res={cat:[] for cat in NEWS_FEEDS}; lock=threading.Lock()
    def fetch_one(cat,name,url):
        try:
            raw=requests.get(url,headers=_UA,timeout=10)
            f=feedparser.parse(raw.content)
            for e in f.entries[:6]:
                t=(e.get('title') or '').strip()
                if not t: continue
                summ=re.sub(r'<[^>]+>','',e.get('summary') or e.get('description') or '')[:260].strip()
                img=None
                try:
                    if getattr(e,'media_thumbnail',None): img=e.media_thumbnail[0].get('url')
                    elif getattr(e,'media_content',None):
                        for m in e.media_content:
                            u=m.get('url','')
                            if u and ('image' in m.get('type','') or u.split('?')[0].lower().endswith(('.jpg','.jpeg','.png','.webp'))): img=u; break
                    elif getattr(e,'enclosures',None):
                        for enc in e.enclosures:
                            if enc.get('type','').startswith('image/'): img=enc.get('url'); break
                except: pass
                with lock: res[cat].append({'src':name,'title':t,'summary':summ,
                                             'link':e.get('link','#'),'published':e.get('published',''),'category':cat,'img':img})
        except: pass

    threads=[]
    for cat,feeds in NEWS_FEEDS.items():
        for name,url in feeds:
            t=threading.Thread(target=fetch_one,args=(cat,name,url),daemon=True); threads.append(t)
    for t in threads: t.start()
    for t in threads: t.join(timeout=12)

    seen=set(); clean={cat:[] for cat in NEWS_FEEDS}
    for cat,arts in res.items():
        for a in arts:
            k=a['title'][:60].lower()
            if k not in seen: seen.add(k); clean[cat].append(a)

    _news_cache['data']=clean; _news_cache['ts']=now
    return [a for arts in clean.values() for a in arts] if category=='all' else clean.get(category,[])

# ── portfolio ──────────────────────────────────────────────────────────────────
def get_portfolio():
    holdings = cfg().get('portfolio', [])
    if not holdings:
        return {'holdings':[],'total_value':0,'total_cost':0,'total_pnl':0,'total_pnl_pct':0,'day_pnl':0}
    res=[]; lock=threading.Lock()
    def fetch(h):
        q=get_quote(h['symbol'])
        if not q: return
        price=q['price']; shares=float(h['shares']); cost=float(h['cost'])
        value=price*shares; invested=cost*shares; pnl=value-invested
        pnl_pct=pnl/invested*100 if invested else 0
        prev=price/(1+q['pct']/100) if q['pct']!=-100 else price
        day_pnl=(price-prev)*shares
        with lock: res.append({
            'symbol':h['symbol'],'shares':shares,'cost':cost,
            'price':round(float(price),2),'value':round(value,2),
            'invested':round(invested,2),'pnl':round(pnl,2),
            'pnl_pct':round(pnl_pct,2),'day_pct':round(float(q['pct']),2),
            'day_pnl':round(day_pnl,2),
        })
    threads=[threading.Thread(target=fetch,args=(h,),daemon=True) for h in holdings]
    for t in threads: t.start()
    for t in threads: t.join(timeout=20)
    res.sort(key=lambda x:x['value'],reverse=True)
    tv=sum(r['value'] for r in res); tc=sum(r['invested'] for r in res)
    for r in res: r['allocation']=round(r['value']/tv*100,1) if tv else 0
    return {'holdings':res,'total_value':round(tv,2),'total_cost':round(tc,2),
            'total_pnl':round(tv-tc,2),'total_pnl_pct':round((tv-tc)/tc*100,2) if tc else 0,
            'day_pnl':round(sum(r['day_pnl'] for r in res),2)}

# ── alerts ──────────────────────────────────────────────────────────────────────
def get_alerts():
    alerts=cfg().get('alerts',[])
    if not alerts: return []
    syms=list({a['symbol'] for a in alerts}); quotes={}; lock=threading.Lock()
    def fq(sym):
        q=get_quote(sym)
        if q:
            with lock: quotes[sym]=q
    threads=[threading.Thread(target=fq,args=(s,),daemon=True) for s in syms]
    for t in threads: t.start()
    for t in threads: t.join(timeout=20)
    res=[]
    for i,a in enumerate(alerts):
        q=quotes.get(a['symbol'])
        if q:
            price=q['price']; target=float(a['target']); cond=a['condition']
            triggered=(cond=='above' and price>=target) or (cond=='below' and price<=target)
            res.append({'id':i,'symbol':a['symbol'],'target':target,'condition':cond,
                        'note':a.get('note',''),'price':round(float(price),2),
                        'pct':round(float(q['pct']),2),'triggered':bool(triggered),
                        'distance':round((price-target)/target*100,2)})
        else:
            res.append({'id':i,'symbol':a['symbol'],'target':float(a['target']),
                        'condition':a['condition'],'note':a.get('note',''),
                        'price':None,'pct':0,'triggered':False,'distance':0})
    res.sort(key=lambda x:(not x['triggered'],abs(x.get('distance',999))))
    return res

# ── economic calendar ───────────────────────────────────────────────────────────
def get_economic_calendar():
    from datetime import date
    today=str(date.today())
    events=[
        # FOMC
        {'date':'2026-06-18','event':'FOMC Rate Decision','type':'fomc','importance':'high','note':'Rate announcement + press conference'},
        {'date':'2026-07-30','event':'FOMC Rate Decision','type':'fomc','importance':'high','note':'Rate announcement'},
        {'date':'2026-09-17','event':'FOMC Rate Decision','type':'fomc','importance':'high','note':'Rate announcement + SEP projections'},
        {'date':'2026-10-29','event':'FOMC Rate Decision','type':'fomc','importance':'high','note':'Rate announcement'},
        {'date':'2026-12-10','event':'FOMC Rate Decision','type':'fomc','importance':'high','note':'Rate announcement + SEP projections'},
        # CPI
        {'date':'2026-06-11','event':'CPI Inflation','type':'cpi','importance':'high','note':'May 2026 — BLS Consumer Price Index'},
        {'date':'2026-07-15','event':'CPI Inflation','type':'cpi','importance':'high','note':'Jun 2026'},
        {'date':'2026-08-12','event':'CPI Inflation','type':'cpi','importance':'high','note':'Jul 2026'},
        {'date':'2026-09-11','event':'CPI Inflation','type':'cpi','importance':'high','note':'Aug 2026'},
        {'date':'2026-10-14','event':'CPI Inflation','type':'cpi','importance':'high','note':'Sep 2026'},
        {'date':'2026-11-12','event':'CPI Inflation','type':'cpi','importance':'high','note':'Oct 2026'},
        {'date':'2026-12-11','event':'CPI Inflation','type':'cpi','importance':'high','note':'Nov 2026'},
        # NFP first Friday
        {'date':'2026-07-02','event':'Nonfarm Payrolls','type':'nfp','importance':'high','note':'Jun 2026 jobs report — BLS'},
        {'date':'2026-08-07','event':'Nonfarm Payrolls','type':'nfp','importance':'high','note':'Jul 2026'},
        {'date':'2026-09-04','event':'Nonfarm Payrolls','type':'nfp','importance':'high','note':'Aug 2026'},
        {'date':'2026-10-02','event':'Nonfarm Payrolls','type':'nfp','importance':'high','note':'Sep 2026'},
        {'date':'2026-11-06','event':'Nonfarm Payrolls','type':'nfp','importance':'high','note':'Oct 2026'},
        {'date':'2026-12-04','event':'Nonfarm Payrolls','type':'nfp','importance':'high','note':'Nov 2026'},
        # PCE last Friday
        {'date':'2026-06-26','event':'PCE Inflation','type':'pce','importance':'high','note':'May 2026 — Fed preferred inflation gauge'},
        {'date':'2026-07-31','event':'PCE Inflation','type':'pce','importance':'high','note':'Jun 2026'},
        {'date':'2026-08-28','event':'PCE Inflation','type':'pce','importance':'high','note':'Jul 2026'},
        {'date':'2026-09-25','event':'PCE Inflation','type':'pce','importance':'high','note':'Aug 2026'},
        {'date':'2026-10-30','event':'PCE Inflation','type':'pce','importance':'high','note':'Sep 2026'},
        {'date':'2026-11-25','event':'PCE Inflation','type':'pce','importance':'high','note':'Oct 2026'},
        # GDP quarterly advance
        {'date':'2026-07-30','event':'GDP Q2 Advance Estimate','type':'gdp','importance':'high','note':'Bureau of Economic Analysis — first read of Q2 2026 growth'},
        {'date':'2026-10-29','event':'GDP Q3 Advance Estimate','type':'gdp','importance':'high','note':'Q3 2026 growth first read'},
        # Retail Sales ~mid-month
        {'date':'2026-06-16','event':'Retail Sales','type':'retail','importance':'medium','note':'May 2026 consumer spending'},
        {'date':'2026-07-16','event':'Retail Sales','type':'retail','importance':'medium','note':'Jun 2026'},
        {'date':'2026-08-14','event':'Retail Sales','type':'retail','importance':'medium','note':'Jul 2026'},
        {'date':'2026-09-16','event':'Retail Sales','type':'retail','importance':'medium','note':'Aug 2026'},
        {'date':'2026-10-15','event':'Retail Sales','type':'retail','importance':'medium','note':'Sep 2026'},
        {'date':'2026-11-17','event':'Retail Sales','type':'retail','importance':'medium','note':'Oct 2026'},
        # ISM Manufacturing first biz day
        {'date':'2026-07-01','event':'ISM Manufacturing PMI','type':'ism','importance':'medium','note':'Jun 2026'},
        {'date':'2026-08-03','event':'ISM Manufacturing PMI','type':'ism','importance':'medium','note':'Jul 2026'},
        {'date':'2026-09-01','event':'ISM Manufacturing PMI','type':'ism','importance':'medium','note':'Aug 2026'},
        {'date':'2026-10-01','event':'ISM Manufacturing PMI','type':'ism','importance':'medium','note':'Sep 2026'},
        {'date':'2026-11-02','event':'ISM Manufacturing PMI','type':'ism','importance':'medium','note':'Oct 2026'},
        {'date':'2026-12-01','event':'ISM Manufacturing PMI','type':'ism','importance':'medium','note':'Nov 2026'},
    ]
    return [e for e in sorted(events,key=lambda x:x['date']) if e['date']>=today]

# ── earnings calendar ──────────────────────────────────────────────────────────
def get_earnings_calendar():
    from datetime import date
    today=str(date.today())
    tickers=list(dict.fromkeys(SCREENER[:30]+cfg().get('watchlist',[])))
    events=[]; lock=threading.Lock()
    def fetch(sym):
        try:
            t=yf.Ticker(sym); cal=t.calendar
            if not isinstance(cal,dict): return
            dates=cal.get('Earnings Date',[])
            if not isinstance(dates,(list,)): dates=[dates]
            for d in dates[:2]:
                ds=d.strftime('%Y-%m-%d') if hasattr(d,'strftime') else str(d)[:10]
                if ds>=today:
                    with lock: events.append({'symbol':sym,'date':ds,'type':'earnings'})
                    return
        except: pass
    threads=[threading.Thread(target=fetch,args=(s,),daemon=True) for s in tickers]
    for t in threads: t.start()
    for t in threads: t.join(timeout=25)
    seen=set(); out=[]
    for e in sorted(events,key=lambda x:x['date']):
        if e['symbol'] not in seen:
            seen.add(e['symbol']); out.append(e)
    return out[:50]

# ── parallel fetch helpers ─────────────────────────────────────────────────────
def parallel_quotes(pairs):
    res={}; lock=threading.Lock()
    def fetch(name,sym):
        q=get_quote(sym)
        with lock: res[name]=q
    threads=[threading.Thread(target=fetch,args=(n,s),daemon=True) for n,s in pairs]
    for t in threads: t.start()
    for t in threads: t.join(timeout=20)
    return res

# ── Flask routes ───────────────────────────────────────────────────────────────
@app.route('/')
def markets(): return render_template_string(MARKETS_HTML)

@app.route('/calendar')
def calendar_page(): return render_template_string(CALENDAR_HTML)

@app.route('/news')
def news_page(): return render_template_string(NEWS_HTML)

@app.route('/api/indices')
def api_indices(): return jsonify(parallel_quotes(INDICES))

@app.route('/api/sectors')
def api_sectors(): return jsonify(parallel_quotes(SECTORS))

@app.route('/api/assets')
def api_assets(): return jsonify(parallel_quotes(ASSETS))

@app.route('/api/screener')
def api_screener():
    tickers=list(dict.fromkeys(SCREENER+cfg().get('watchlist',[])))
    res=[]; lock=threading.Lock()
    def fetch(sym):
        r=get_analysis(sym)
        if r:
            with lock: res.append(r)
    threads=[threading.Thread(target=fetch,args=(s,),daemon=True) for s in tickers]
    for t in threads: t.start()
    for t in threads: t.join(timeout=60)
    res.sort(key=lambda x:abs(x['pct']),reverse=True)
    return jsonify(res)

@app.route('/api/lookup')
def api_lookup():
    sym=request.args.get('symbol','').strip().upper()
    if not sym: return jsonify({'error':'empty'}),400
    r=get_analysis(sym)
    if not r: return jsonify({'error':f'No data for {sym}'}),404
    return jsonify(r)

@app.route('/api/news')
def api_news():
    cat=request.args.get('category','all')
    return jsonify(get_news(cat))

@app.route('/api/refresh')
def api_refresh():
    _news_cache['ts']=0
    return jsonify({'ok':True})


@app.route('/api/alerts', methods=['GET','POST'])
def api_alerts():
    if request.method=='POST':
        save_cfg({'alerts': request.json or []}); return jsonify({'ok':True})
    return jsonify(get_alerts())

@app.route('/api/calendar/earnings')
def api_earnings(): return jsonify(get_earnings_calendar())

@app.route('/api/calendar/economic')
def api_economic(): return jsonify(get_economic_calendar())


@app.route('/api/config', methods=['GET','POST'])
def api_config():
    if request.method=='POST':
        save_cfg(request.json or {}); return jsonify({'ok':True})
    return jsonify(cfg())

@app.route('/api/digest')
def api_digest():
    """Computed daily digest — no AI, no API key. Summarizes markets + news."""
    def pack(quotes):
        out=[]
        for name,d in quotes.items():
            if d and d.get('pct') is not None:
                out.append({'name':name,'pct':round(d['pct'],2),'price':d.get('price')})
        return out
    indices=pack(parallel_quotes(INDICES))
    sectors=sorted(pack(parallel_quotes(SECTORS)),key=lambda x:x['pct'],reverse=True)
    assets =sorted(pack(parallel_quotes(ASSETS)), key=lambda x:x['pct'],reverse=True)
    up   =sum(1 for s in sectors if s['pct']>0)
    down =sum(1 for s in sectors if s['pct']<0)
    movers=sorted(sectors+assets,key=lambda x:abs(x['pct']),reverse=True)[:6]
    spx=next((i['pct'] for i in indices if 'S&P' in i['name']), None)
    news=get_news('all')
    cats={}
    for a in news: cats.setdefault(a['category'],[]).append(a)
    news_by_cat=sorted(
        [{'category':c,'count':len(arts),
          'top':[{'title':x['title'],'src':x['src'],'link':x['link']} for x in arts[:3]]}
         for c,arts in cats.items()],
        key=lambda x:x['count'],reverse=True)
    headlines=[{'title':x['title'],'src':x['src'],'link':x['link'],'category':x['category']} for x in news[:8]]
    return jsonify({
        'generated_at':datetime.now().strftime('%I:%M %p'),
        'date':datetime.now().strftime('%A, %B %d, %Y'),
        'spx':spx,
        'breadth':{'up':up,'down':down,'total':len(sectors)},
        'indices':indices,
        'sectors_top':sectors[:3],
        'sectors_bottom':sectors[-3:][::-1],
        'movers':movers,
        'news_by_cat':news_by_cat,
        'headlines':headlines,
        'news_total':len(news),
    })



# ── CSS ────────────────────────────────────────────────────────────────────────
CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

:root{
  --bg:#06080f;--bg2:#0c1120;--bg3:#121929;--border:#1a2640;
  --text:#e2eaff;--muted:#5a7199;
  --pos:#00f090;--neg:#ff3d6b;--warn:#ffb340;
  --acc:#4d9fff;--acc2:#c471f5;--acc3:#00d4aa;
  --glow-pos:rgba(0,240,144,.4);--glow-neg:rgba(255,61,107,.4);--glow-acc:rgba(77,159,255,.35);
}
*{box-sizing:border-box;margin:0;padding:0;}
html{scroll-behavior:smooth;}
body{
  background:var(--bg);
  background-image:radial-gradient(ellipse at 20% 0%,rgba(77,159,255,.07) 0%,transparent 60%),
                   radial-gradient(ellipse at 80% 100%,rgba(196,113,245,.06) 0%,transparent 60%);
  background-attachment:fixed;
  color:var(--text);font-family:'Inter',-apple-system,'Segoe UI',sans-serif;font-size:14px;
  min-height:100vh;
}
a{color:var(--acc);text-decoration:none;}a:hover{color:#7ab8ff;text-decoration:underline;}
::-webkit-scrollbar{width:6px;height:6px;}
::-webkit-scrollbar-track{background:var(--bg2);}
::-webkit-scrollbar-thumb{background:var(--border);border-radius:3px;}
::-webkit-scrollbar-thumb:hover{background:var(--muted);}

/* NAV */
.nav{
  background:linear-gradient(90deg,rgba(6,8,15,.97),rgba(12,17,32,.97));
  backdrop-filter:blur(20px);-webkit-backdrop-filter:blur(20px);
  border-bottom:1px solid rgba(77,159,255,.15);
  padding:0 28px;display:flex;align-items:center;height:56px;gap:6px;
  position:sticky;top:0;z-index:100;
  box-shadow:0 1px 30px rgba(0,0,0,.5);
}
.brand{
  font-weight:700;font-size:18px;margin-right:24px;letter-spacing:-.5px;
  background:linear-gradient(135deg,#4d9fff,#c471f5);
  -webkit-background-clip:text;-webkit-text-fill-color:transparent;background-clip:text;
}
.nav-link{
  padding:7px 16px;border-radius:8px;color:var(--muted);font-weight:500;
  transition:all .2s;font-size:13.5px;letter-spacing:.1px;
}
.nav-link:hover{background:rgba(77,159,255,.1);color:var(--text);text-decoration:none;}
.nav-link.active{
  background:linear-gradient(135deg,rgba(77,159,255,.25),rgba(196,113,245,.15));
  color:var(--acc);border:1px solid rgba(77,159,255,.3);
  box-shadow:0 0 16px rgba(77,159,255,.15);
}
.nav-time{margin-left:auto;color:var(--muted);font-size:12px;font-variant-numeric:tabular-nums;}

/* LAYOUT */
.wrap{max-width:1640px;margin:0 auto;padding:24px 28px;}
.sec{margin-bottom:32px;}
.sec-title{
  font-size:10.5px;font-weight:700;text-transform:uppercase;letter-spacing:1.5px;
  color:var(--acc);margin-bottom:14px;
  display:flex;align-items:center;gap:8px;
}
.sec-title::after{content:'';flex:1;height:1px;background:linear-gradient(90deg,rgba(77,159,255,.3),transparent);}

/* CARDS */
.card-row{display:grid;gap:12px;}
.card{
  background:linear-gradient(135deg,rgba(12,17,32,.9),rgba(18,25,41,.9));
  border:1px solid var(--border);border-radius:12px;padding:16px 18px;
  transition:all .2s;position:relative;overflow:hidden;
}
.card::before{
  content:'';position:absolute;top:0;left:0;right:0;height:1px;
  background:linear-gradient(90deg,transparent,rgba(77,159,255,.3),transparent);
}
.card:hover{border-color:rgba(77,159,255,.3);box-shadow:0 4px 24px rgba(0,0,0,.4),0 0 0 1px rgba(77,159,255,.1);}
.card-lbl{font-size:10.5px;color:var(--muted);text-transform:uppercase;letter-spacing:.8px;margin-bottom:6px;font-weight:600;}
.card-price{font-size:20px;font-weight:700;letter-spacing:-.5px;font-variant-numeric:tabular-nums;}
.card-chg{font-size:13px;font-weight:600;margin-top:3px;}

/* VALUE COLORS */
.pos{color:var(--pos);text-shadow:0 0 12px var(--glow-pos);}
.neg{color:var(--neg);text-shadow:0 0 12px var(--glow-neg);}
.neu{color:var(--muted);}
.badge{display:inline-block;padding:3px 9px;border-radius:5px;font-size:11px;font-weight:700;letter-spacing:.3px;}

/* TABLE */
.tbl-wrap{
  overflow-x:auto;border-radius:12px;
  border:1px solid var(--border);
  box-shadow:0 4px 30px rgba(0,0,0,.4);
}
table{width:100%;border-collapse:collapse;}
th{
  background:linear-gradient(180deg,rgba(18,25,41,1),rgba(12,17,32,1));
  color:var(--muted);font-size:10.5px;font-weight:700;text-transform:uppercase;
  letter-spacing:.8px;padding:11px 13px;text-align:right;white-space:nowrap;
  border-bottom:1px solid rgba(77,159,255,.12);
}
th:first-child{text-align:left;}
td{
  padding:10px 13px;border-top:1px solid rgba(255,255,255,.03);
  text-align:right;white-space:nowrap;vertical-align:middle;
  font-variant-numeric:tabular-nums;
}
td:first-child{text-align:left;font-weight:700;font-family:'SF Mono','Fira Code',monospace;font-size:13px;color:var(--text);}
tr:hover td{background:rgba(77,159,255,.05);}
tr.flagged{background:rgba(77,159,255,.03);}
tr.flagged td:first-child{
  border-left:2px solid var(--acc);
  color:var(--acc);text-shadow:0 0 8px var(--glow-acc);
}

/* FILTER BUTTONS */
.filters{display:flex;gap:8px;margin-bottom:16px;flex-wrap:wrap;}
.fbtn{
  padding:6px 16px;border-radius:20px;
  border:1px solid rgba(90,113,153,.4);background:transparent;
  color:var(--muted);font-size:12px;cursor:pointer;transition:all .2s;font-family:inherit;font-weight:500;
}
.fbtn:hover{border-color:var(--acc);color:var(--acc);background:rgba(77,159,255,.08);}
.fbtn.on{
  background:linear-gradient(135deg,rgba(77,159,255,.25),rgba(196,113,245,.15));
  border-color:rgba(77,159,255,.5);color:var(--acc);
  box-shadow:0 0 12px rgba(77,159,255,.2);
}

/* SECTOR GRID */
.sector-grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(130px,1fr));gap:10px;}
.sc{
  border:1px solid var(--border);border-radius:10px;padding:12px 14px;
  transition:all .2s;cursor:default;
}
.sc:hover{transform:translateY(-1px);box-shadow:0 6px 20px rgba(0,0,0,.3);}
.sc-name{font-size:10.5px;color:var(--muted);margin-bottom:5px;font-weight:600;letter-spacing:.3px;}
.sc-pct{font-size:17px;font-weight:700;font-variant-numeric:tabular-nums;}

/* 52W BAR */
.bar-w{display:flex;align-items:center;gap:6px;}
.bar52{width:58px;height:5px;background:rgba(255,255,255,.07);border-radius:3px;overflow:hidden;flex-shrink:0;}
.bar52f{height:100%;border-radius:3px;transition:width .4s;}
.bar52p{font-size:10px;color:var(--muted);width:28px;text-align:right;font-variant-numeric:tabular-nums;}

/* NEWS CATEGORY TABS */
.cat-tabs{
  display:flex;gap:0;border-bottom:1px solid var(--border);margin-bottom:22px;flex-wrap:wrap;
}
.ctab{
  padding:11px 20px;color:var(--muted);font-size:13px;font-weight:500;cursor:pointer;
  border-bottom:2px solid transparent;margin-bottom:-1px;transition:all .2s;
}
.ctab:hover{color:var(--text);}
.ctab.on{color:var(--acc);border-bottom-color:var(--acc);text-shadow:0 0 10px var(--glow-acc);}

/* NEWS CARDS */
.news-grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(340px,1fr));gap:16px;}
.nc{
  background:linear-gradient(135deg,rgba(12,17,32,.95),rgba(18,25,41,.95));
  border:1px solid var(--border);border-radius:12px;
  padding:16px 18px;display:flex;flex-direction:column;gap:9px;
  transition:all .2s;position:relative;overflow:hidden;
}
.nc:hover{border-color:rgba(77,159,255,.25);box-shadow:0 6px 30px rgba(0,0,0,.4);transform:translateY(-1px);}
.nc-Global   {border-left:3px solid var(--acc);}
.nc-Geopolitics{border-left:3px solid var(--acc2);}
.nc-Policy   {border-left:3px solid var(--warn);}
.nc-NatSecurity{border-left:3px solid var(--neg);}
.nc-Economy  {border-left:3px solid var(--pos);}
.nc-top{display:flex;align-items:center;gap:6px;flex-wrap:wrap;}
.ns-badge{
  background:rgba(255,255,255,.06);color:var(--muted);font-size:10px;font-weight:700;
  text-transform:uppercase;letter-spacing:.7px;padding:2px 7px;border-radius:4px;
}
.nc-cat{font-size:10px;font-weight:700;padding:2px 7px;border-radius:4px;
        text-transform:uppercase;letter-spacing:.5px;}
.cat-Global     {background:rgba(77,159,255,.15);color:var(--acc);}
.cat-Geopolitics{background:rgba(196,113,245,.15);color:var(--acc2);}
.cat-Policy     {background:rgba(255,179,64,.15);color:var(--warn);}
.cat-NatSecurity{background:rgba(255,61,107,.15);color:var(--neg);}
.cat-Economy    {background:rgba(0,240,144,.12);color:var(--pos);}
.nc-time{font-size:11px;color:rgba(255,255,255,.2);margin-left:auto;}
.nc-title{font-size:14px;font-weight:600;line-height:1.45;}
.nc-title a{color:var(--text);}
.nc-title a:hover{color:var(--acc);}
.nc-summ{font-size:12px;color:var(--muted);line-height:1.6;}

/* CHAT */
.chat-wrap{display:flex;flex-direction:column;height:calc(100vh - 210px);max-width:880px;margin:0 auto;}
.chat-hist{
  flex:1;overflow-y:auto;display:flex;flex-direction:column;gap:14px;padding:18px;
  background:linear-gradient(180deg,rgba(12,17,32,.9),rgba(6,8,15,.9));
  border:1px solid var(--border);border-bottom:none;border-radius:12px 12px 0 0;
}
.msg{display:flex;flex-direction:column;gap:4px;}
.msg.user .bubble{
  background:linear-gradient(135deg,#3a8fff,#6b4fff);color:#fff;
  align-self:flex-end;border-radius:16px 16px 4px 16px;max-width:75%;
  box-shadow:0 4px 16px rgba(77,159,255,.3);
}
.msg.assistant .bubble{
  background:rgba(18,25,41,.8);border:1px solid var(--border);
  border-radius:16px 16px 16px 4px;max-width:85%;
}
.bubble{padding:11px 15px;font-size:13.5px;line-height:1.65;white-space:pre-wrap;}
.role{font-size:10px;font-weight:700;text-transform:uppercase;letter-spacing:.8px;color:var(--muted);padding:0 4px;}
.msg.user .role{text-align:right;}
.input-row{
  display:flex;gap:10px;
  background:rgba(12,17,32,.95);padding:14px;
  border:1px solid var(--border);border-top:none;border-radius:0 0 12px 12px;
}
.tinput{
  flex:1;background:rgba(18,25,41,.9);border:1px solid var(--border);color:var(--text);
  border-radius:10px;padding:11px 15px;font-size:13px;font-family:inherit;resize:none;outline:none;
  transition:border-color .2s;
}
.tinput:focus{border-color:rgba(77,159,255,.5);box-shadow:0 0 0 3px rgba(77,159,255,.1);}
.btn{
  padding:11px 22px;border-radius:10px;border:none;cursor:pointer;font-size:13px;
  font-weight:700;font-family:inherit;transition:all .2s;white-space:nowrap;letter-spacing:.2px;
}
.btn-p{
  background:linear-gradient(135deg,#4d9fff,#6b4fff);color:#fff;
  box-shadow:0 4px 14px rgba(77,159,255,.35);
}
.btn-p:hover{box-shadow:0 6px 20px rgba(77,159,255,.5);transform:translateY(-1px);}
.btn-p:disabled{opacity:.5;transform:none;box-shadow:none;}
.btn-g{background:rgba(255,255,255,.05);border:1px solid var(--border);color:var(--muted);}
.btn-g:hover{border-color:rgba(77,159,255,.4);color:var(--acc);background:rgba(77,159,255,.08);}

/* INPUTS */
.key-row{
  display:flex;gap:10px;align-items:center;padding:12px 16px;
  background:linear-gradient(135deg,rgba(12,17,32,.9),rgba(18,25,41,.9));
  border:1px solid var(--border);border-radius:12px;margin-bottom:14px;
}
.key-row label{font-size:12px;color:var(--muted);white-space:nowrap;font-weight:600;}
.kinput{
  flex:1;background:rgba(6,8,15,.8);border:1px solid var(--border);color:var(--text);
  border-radius:8px;padding:8px 13px;font-size:12px;font-family:'SF Mono','Fira Code',monospace;outline:none;
  transition:border-color .2s;
}
.kinput:focus{border-color:rgba(77,159,255,.5);}
.suggs{display:flex;flex-wrap:wrap;gap:8px;margin-bottom:14px;}
.sugg{
  padding:7px 14px;
  background:linear-gradient(135deg,rgba(12,17,32,.9),rgba(18,25,41,.9));
  border:1px solid rgba(77,159,255,.2);border-radius:20px;
  font-size:12px;color:var(--muted);cursor:pointer;transition:all .2s;font-weight:500;
}
.sugg:hover{border-color:var(--acc);color:var(--acc);background:rgba(77,159,255,.1);box-shadow:0 0 12px rgba(77,159,255,.15);}

/* WATCHLIST ROW */
.watchlist-row{
  display:flex;gap:8px;align-items:center;padding:10px 14px;
  background:rgba(12,17,32,.8);border:1px solid var(--border);border-radius:10px;margin-bottom:12px;
}
.watchlist-row label{font-size:12px;color:var(--muted);white-space:nowrap;font-weight:600;}
.winput{
  flex:1;background:rgba(6,8,15,.8);border:1px solid var(--border);color:var(--text);
  border-radius:7px;padding:7px 12px;font-size:13px;font-family:'SF Mono','Fira Code',monospace;outline:none;
}
.winput:focus{border-color:rgba(77,159,255,.4);}
.loading{color:var(--muted);font-size:13px;padding:28px;text-align:center;letter-spacing:.2px;}

/* SUMMARY CARDS ROW */
.sum-row{display:grid;gap:12px;margin-bottom:24px;}
.sum-card{background:linear-gradient(135deg,rgba(12,17,32,.9),rgba(18,25,41,.9));
  border:1px solid var(--border);border-radius:12px;padding:18px 20px;position:relative;overflow:hidden;}
.sum-card::before{content:'';position:absolute;top:0;left:0;right:0;height:2px;
  background:linear-gradient(90deg,var(--acc),var(--acc2));}
.sum-label{font-size:11px;color:var(--muted);font-weight:700;text-transform:uppercase;letter-spacing:1px;margin-bottom:8px;}
.sum-val{font-size:26px;font-weight:700;letter-spacing:-.5px;font-variant-numeric:tabular-nums;}
.sum-sub{font-size:13px;font-weight:600;margin-top:4px;}

/* PORTFOLIO TABLE */
.alloc-bar{height:6px;border-radius:3px;background:linear-gradient(90deg,var(--acc),var(--acc2));margin-top:4px;}

/* ALERTS */
.alert-triggered{background:rgba(0,240,144,.06);border-color:rgba(0,240,144,.3);}
.alert-triggered.neg-alert{background:rgba(255,61,107,.06);border-color:rgba(255,61,107,.3);}
.alert-card{display:flex;align-items:center;gap:12px;padding:12px 16px;
  border-radius:10px;border:1px solid var(--border);background:rgba(12,17,32,.8);margin-bottom:8px;}
.alert-sym{font-size:14px;font-weight:700;font-family:'SF Mono',monospace;min-width:60px;}
.alert-status{padding:3px 10px;border-radius:20px;font-size:11px;font-weight:700;letter-spacing:.5px;}
.status-triggered{background:rgba(0,240,144,.15);color:var(--pos);}
.status-watching{background:rgba(90,113,153,.15);color:var(--muted);}

/* CALENDAR */
.cal-event{display:flex;gap:14px;padding:12px 16px;border-radius:10px;
  border:1px solid var(--border);background:rgba(12,17,32,.8);margin-bottom:8px;align-items:flex-start;}
.cal-date-col{min-width:90px;flex-shrink:0;}
.cal-date{font-size:14px;font-weight:700;font-variant-numeric:tabular-nums;}
.cal-dow{font-size:10px;color:var(--muted);font-weight:600;text-transform:uppercase;letter-spacing:.5px;}
.cal-type{padding:2px 8px;border-radius:4px;font-size:10px;font-weight:800;letter-spacing:.8px;text-transform:uppercase;flex-shrink:0;}
.type-fomc{background:rgba(196,113,245,.18);color:var(--acc2);}
.type-cpi,.type-pce{background:rgba(255,179,64,.15);color:var(--warn);}
.type-nfp{background:rgba(0,240,144,.12);color:var(--pos);}
.type-gdp{background:rgba(77,159,255,.15);color:var(--acc);}
.type-retail,.type-ism{background:rgba(139,148,158,.12);color:var(--muted);}
.type-earnings{background:rgba(77,159,255,.12);color:var(--acc);}
.cal-name{font-size:13.5px;font-weight:600;}
.cal-note{font-size:11.5px;color:var(--muted);margin-top:3px;}
.cal-upcoming{border-left:2px solid var(--acc);}
.cal-today{border-left:2px solid var(--pos);background:rgba(0,240,144,.04);}

/* FORM */
.form-row{display:flex;gap:10px;flex-wrap:wrap;align-items:flex-end;
  background:rgba(12,17,32,.8);border:1px solid var(--border);border-radius:12px;padding:16px;}
.form-field{display:flex;flex-direction:column;gap:5px;flex:1;min-width:100px;}
.form-field label{font-size:11px;color:var(--muted);font-weight:600;text-transform:uppercase;letter-spacing:.5px;}
.form-input{background:rgba(6,8,15,.9);border:1px solid var(--border);color:var(--text);
  border-radius:8px;padding:9px 12px;font-size:13px;font-family:inherit;outline:none;width:100%;}
.form-input:focus{border-color:rgba(77,159,255,.5);}
select.form-input{cursor:pointer;}

/* BRIEFING */
.briefing-card{background:linear-gradient(135deg,rgba(12,17,32,.95),rgba(18,25,41,.95));
  border:1px solid rgba(77,159,255,.2);border-radius:12px;padding:20px 24px;
  font-size:13.5px;line-height:1.75;color:var(--text);white-space:pre-wrap;
  box-shadow:0 0 30px rgba(77,159,255,.08);}

/* TICKER */
.ticker-bar{
  background:rgba(4,6,12,.98);border-bottom:1px solid rgba(77,159,255,.1);
  height:28px;display:flex;align-items:center;overflow:hidden;
  position:sticky;top:56px;z-index:98;
}
.ticker-tag{
  background:linear-gradient(135deg,#4d9fff,#7b4fff);
  color:#fff;font-size:9.5px;font-weight:800;letter-spacing:1.8px;
  padding:0 14px;height:100%;display:flex;align-items:center;
  flex-shrink:0;text-transform:uppercase;gap:5px;
}
.ticker-dot-live{
  width:5px;height:5px;border-radius:50%;background:#fff;
  animation:blink .9s ease-in-out infinite;
}
@keyframes blink{0%,100%{opacity:1;}50%{opacity:.25;}}
.ticker-outer{flex:1;overflow:hidden;position:relative;height:100%;cursor:default;}
.ticker-inner{display:inline-flex;align-items:center;height:100%;white-space:nowrap;will-change:transform;}
</style>
"""

NAV = """
<nav class="nav">
  <span class="brand">Daily<span>Pulse</span></span>
  <a href="/" class="nav-link {MA}">Markets</a>
  <a href="/digest" class="nav-link {LA}">Digest</a>
  <a href="/calendar" class="nav-link {CA}">Calendar</a>
  <a href="/news" class="nav-link {NA}">News</a>
  <span class="nav-time" id="clk"></span>
</nav>
<div class="ticker-bar">
  <div class="ticker-tag"><div class="ticker-dot-live"></div>LIVE</div>
  <div class="ticker-outer" id="tick-outer">
    <div class="ticker-inner" id="tick-inner"></div>
  </div>
</div>
<script>
(function(){
  // clock
  function tickClock(){
    document.getElementById('clk').textContent=
      new Date().toLocaleDateString('en-US',{weekday:'short',month:'short',day:'numeric'})+
      '  '+new Date().toLocaleTimeString('en-US',{hour:'2-digit',minute:'2-digit'});
  }
  tickClock(); setInterval(tickClock,30000);

  // news ticker
  let tx=0, half=0, paused=false;
  const CATS={'Global':'#4d9fff','Geopolitics':'#c471f5','Policy':'#ffb340',
              'Nat. Security':'#ff3d6b','Economy':'#00f090'};

  async function fetchTicker(){
    const data=await fetch('/api/news').then(r=>r.json()).catch(()=>null);
    if(!data||!data.length)return;
    const items=data.slice(0,35).map(a=>{
      const c=CATS[a.category]||'#5a7199';
      const bg=c+'22';
      return '<span style="display:inline-flex;align-items:center;gap:7px;padding:0 6px">'
        +'<span style="font-size:9.5px;font-weight:800;color:'+c+';background:'+bg
        +';padding:1px 6px;border-radius:3px;text-transform:uppercase;letter-spacing:.5px;flex-shrink:0">'
        +a.src+'</span>'
        +'<span style="font-size:12px;color:rgba(226,234,255,.88)">'+a.title+'</span>'
        +'</span>'
        +'<span style="color:rgba(77,159,255,.3);margin:0 16px;font-size:9px;flex-shrink:0">&#9670;</span>';
    }).join('');
    const el=document.getElementById('tick-inner');
    el.innerHTML=items+items;
    half=el.scrollWidth/2;
    if(tx<-half||tx>0)tx=0;
  }

  function animTicker(){
    if(!paused){
      tx-=0.55;
      if(half>0&&tx<=-half)tx=0;
      const el=document.getElementById('tick-inner');
      if(el)el.style.transform='translateX('+tx+'px)';
    }
    requestAnimationFrame(animTicker);
  }

  const outer=document.getElementById('tick-outer');
  outer.addEventListener('mouseenter',()=>paused=true);
  outer.addEventListener('mouseleave',()=>paused=false);
  fetchTicker();
  requestAnimationFrame(animTicker);
  setInterval(fetchTicker,600000);
})();
</script>
"""

# ── Markets page ───────────────────────────────────────────────────────────────
MARKETS_HTML = """<!DOCTYPE html><html lang="en"><head>
<meta charset="UTF-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Markets — DailyPulse</title>""" + CSS + """</head><body>
""" + NAV.replace('{MA}','active').replace('{PA}','').replace('{LA}','').replace('{CA}','').replace('{NA}','').replace('{AA}','').replace('{DA}','') + """
<div class="wrap">
<div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:18px;flex-wrap:wrap;gap:8px;">
  <div style="font-size:12px;color:var(--muted)">Updated: <span id="upd-time" style="color:var(--acc)">loading…</span></div>
  <div style="display:flex;gap:8px">
    <button class="fbtn" onclick="refreshAll()" id="refresh-btn" style="font-size:11px;padding:4px 12px">&#8635; Refresh All</button>
  </div>
</div>

<div class="sec">
  <div class="sec-title">Market Indices</div>
  <div class="card-row" style="grid-template-columns:repeat(auto-fill,minmax(175px,1fr))" id="idx"></div>
</div>

<div class="sec">
  <div class="sec-title">Sector Performance</div>
  <div class="sector-grid" id="sectors"></div>
</div>

<div class="sec">
  <div class="sec-title">Commodities &amp; Crypto</div>
  <div class="card-row" style="grid-template-columns:repeat(auto-fill,minmax(145px,1fr))" id="assets"></div>
</div>

<div class="sec">
  <div class="sec-title">Investment Screener &amp; Analysis</div>
  <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:10px;flex-wrap:wrap;gap:8px;">
    <div class="filters" style="margin-bottom:0">
      <button class="fbtn on" data-f="all">All</button>
      <button class="fbtn" data-f="movers">Big Movers (&gt;2%)</button>
      <button class="fbtn" data-f="bullish">Bullish Signal</button>
      <button class="fbtn" data-f="bearish">Bearish Signal</button>
      <button class="fbtn" data-f="oversold">Oversold RSI</button>
      <button class="fbtn" data-f="highvol">High Volume</button>
      <button class="fbtn" data-f="near52h">Near 52W High</button>
    </div>
    <div class="watchlist-row" style="margin-bottom:0;padding:6px 10px;">
      <label>Watchlist:</label>
      <input class="winput" id="wl-input" placeholder="AAPL,TSLA,NVDA" style="width:220px">
      <button class="btn btn-g" style="padding:6px 12px;font-size:12px" onclick="saveWatchlist()">Save &amp; Refresh</button>
    </div>
  </div>
  <div style="display:flex;gap:8px;align-items:center;margin-bottom:10px;flex-wrap:wrap">
    <input class="winput" id="sc-search" placeholder="Search ticker — e.g. AAPL" style="width:240px">
    <button class="btn btn-g" style="padding:6px 12px;font-size:12px" onclick="searchTicker()">Search</button>
    <span id="sc-search-note" style="font-size:12px;color:var(--muted)"></span>
  </div>
  <div class="tbl-wrap">
    <table>
      <thead><tr>
        <th>Symbol</th><th>Price</th><th>1D%</th><th>5D%</th>
        <th>52W Range</th><th>RSI</th><th>MACD</th>
        <th>Vol/Avg</th><th>Beta</th><th>Volatility</th><th>Signal</th>
      </tr></thead>
      <tbody id="scrn"><tr><td colspan="11" class="loading">Loading — may take 30–60 sec…</td></tr></tbody>
    </table>
  </div>
</div>
</div>

<script>
let scData=[], activeF='all';

function fp(v){
  if(v==null)return '—';
  return v>=1000?'$'+v.toLocaleString('en',{minimumFractionDigits:2,maximumFractionDigits:2}):
         v>=1?'$'+v.toFixed(2):'$'+v.toFixed(4);
}
function fpct(v){
  if(v==null)return '—';
  const s=v>=0?'+':'';
  return `<span class="${v>=0?'pos':'neg'}">${s}${v.toFixed(2)}%</span>`;
}
function bar52(p){
  const w=Math.min(100,Math.max(0,p));
  const c=p>80?'#3fb950':p<20?'#f85149':'#58a6ff';
  return `<div class="bar-w"><div class="bar52"><div class="bar52f" style="width:${w}%;background:${c}"></div></div><span class="bar52p">${p.toFixed(0)}%</span></div>`;
}
function rsiCls(r){return r==null?'neu':r<30?'pos':r>70?'neg':'neu';}

async function loadIndices(){
  const r=await fetch('/api/indices').then(r=>r.json()).catch(()=>null); if(!r)return;
  document.getElementById('idx').innerHTML=Object.entries(r).map(([n,d])=>{
    if(!d)return`<div class="card"><div class="card-lbl">${n}</div><div class="card-price neu">—</div></div>`;
    const c=d.pct>=0?'pos':'neg',s=d.pct>=0?'+':'';
    return`<div class="card"><div class="card-lbl">${n}</div>
      <div class="card-price">${fp(d.price).replace('$','')}</div>
      <div class="card-chg ${c}">${s}${d.pct.toFixed(2)}% <span style="font-weight:400;font-size:12px">${s}${d.change.toFixed(2)}</span></div>
    </div>`;
  }).join('');
}
async function loadSectors(){
  const r=await fetch('/api/sectors').then(r=>r.json()).catch(()=>null); if(!r)return;
  document.getElementById('sectors').innerHTML=Object.entries(r).map(([n,d])=>{
    if(!d)return`<div class="sc"><div class="sc-name">${n}</div><div class="sc-pct neu">—</div></div>`;
    const c=d.pct>=0?'pos':'neg',s=d.pct>=0?'+':'';
    const op=Math.min(0.45,Math.abs(d.pct)/2.5*0.45);
    const glow=d.pct>=0?`0 0 20px rgba(0,240,144,${op*0.6})`:`0 0 20px rgba(255,61,107,${op*0.6})`;
    const bg=d.pct>=0?`rgba(0,240,144,${op*0.18})`:`rgba(255,61,107,${op*0.18})`;
    const border=d.pct>=0?`rgba(0,240,144,${op*0.5})`:`rgba(255,61,107,${op*0.5})`;
    return`<div class="sc" style="background:${bg};border-color:${border};box-shadow:${glow}"><div class="sc-name">${n}</div><div class="sc-pct ${c}">${s}${d.pct.toFixed(2)}%</div></div>`;
  }).join('');
}
async function loadAssets(){
  const r=await fetch('/api/assets').then(r=>r.json()).catch(()=>null); if(!r)return;
  document.getElementById('assets').innerHTML=Object.entries(r).map(([n,d])=>{
    if(!d)return`<div class="card"><div class="card-lbl">${n}</div><div class="card-price">—</div></div>`;
    const c=d.pct>=0?'pos':'neg',s=d.pct>=0?'+':'';
    return`<div class="card"><div class="card-lbl">${n}</div>
      <div class="card-price" style="font-size:16px">${fp(d.price)}</div>
      <div class="card-chg ${c}">${s}${d.pct.toFixed(2)}%</div>
    </div>`;
  }).join('');
}
function renderScrn(data){
  const tb=document.getElementById('scrn');
  if(!data.length){tb.innerHTML='<tr><td colspan="11" class="loading">No results</td></tr>';return;}
  tb.innerHTML=data.map(d=>{
    const m=d.macd||{},sig=d.signal||{};
    const mTxt=m.cross==='bullish'?'<span class="pos">↑ BullX</span>':
                m.cross==='bearish'?'<span class="neg">↓ BearX</span>':
                m.cross==='above'?'<span class="pos">Above</span>':'<span class="neg">Below</span>';
    return`<tr class="${d.flag?'flagged':''}">
      <td>${d.symbol}</td>
      <td>${fp(d.price)}</td>
      <td>${fpct(d.pct)}</td>
      <td>${fpct(d.mom5)}</td>
      <td>${bar52(d.pct52)}</td>
      <td><span class="${rsiCls(d.rsi)}">${d.rsi??'—'}</span></td>
      <td>${mTxt}</td>
      <td><span class="${d.vol_r>2?'pos':'neu'}">${d.vol_r.toFixed(1)}x</span></td>
      <td>${d.beta??'—'}</td>
      <td>${d.vol30!=null?d.vol30+'%':'—'}</td>
      <td><span class="badge" style="background:${sig.color}22;color:${sig.color}">${sig.label}</span></td>
    </tr>`;
  }).join('');
}
async function searchTicker(){
  const inp=document.getElementById('sc-search');
  const note=document.getElementById('sc-search-note');
  const sym=(inp.value||'').trim().toUpperCase();
  if(!sym){return;}
  note.textContent='Searching '+sym+'…';
  let res;
  try{ res=await fetch('/api/lookup?symbol='+encodeURIComponent(sym)); }
  catch(e){ note.innerHTML='<span style="color:var(--neg)">Lookup failed</span>'; return; }
  if(!res.ok){ note.innerHTML='<span style="color:var(--neg)">No data for '+sym+'</span>'; return; }
  const d=await res.json();
  note.textContent=''; inp.value='';
  scData=[d, ...scData.filter(x=>x.symbol!==d.symbol)];
  applyFilter('all');
  const tb=document.getElementById('scrn');
  if(tb&&tb.firstElementChild){tb.firstElementChild.style.outline='2px solid var(--acc)';}
}
function applyFilter(f){
  activeF=f;
  document.querySelectorAll('.fbtn').forEach(b=>b.classList.toggle('on',b.dataset.f===f));
  let d=scData;
  if(f==='movers')  d=d.filter(x=>Math.abs(x.pct)>2);
  if(f==='bullish') d=d.filter(x=>x.signal&&x.signal.score>=1);
  if(f==='bearish') d=d.filter(x=>x.signal&&x.signal.score<=-1);
  if(f==='oversold')d=d.filter(x=>x.rsi!=null&&x.rsi<35);
  if(f==='highvol') d=d.filter(x=>x.vol_r>1.8);
  if(f==='near52h') d=d.filter(x=>x.pct52>88);
  renderScrn(d);
}
document.querySelectorAll('.fbtn[data-f]').forEach(b=>b.addEventListener('click',()=>applyFilter(b.dataset.f)));
(function(){const wl=document.getElementById('wl-input');if(wl)wl.addEventListener('keydown',e=>{if(e.key==='Enter'){e.preventDefault();saveWatchlist();}});})();
(function(){const s=document.getElementById('sc-search');if(s)s.addEventListener('keydown',e=>{if(e.key==='Enter'){e.preventDefault();searchTicker();}});})();

async function loadScreener(){
  const r=await fetch('/api/screener').then(r=>r.json()).catch(()=>null);
  if(!r){document.getElementById('scrn').innerHTML='<tr><td colspan="11" class="loading">Failed to load</td></tr>';return;}
  scData=r; applyFilter(activeF);
}

async function saveWatchlist(){
  const raw=document.getElementById('wl-input').value.trim();
  if(!raw)return;
  const tickers=raw.toUpperCase().split(/[,\\s]+/).filter(Boolean);
  await fetch('/api/config',{method:'POST',headers:{'Content-Type':'application/json'},
    body:JSON.stringify({watchlist:tickers})});
  scData=[]; document.getElementById('scrn').innerHTML='<tr><td colspan="11" class="loading">Refreshing…</td></tr>';
  loadScreener();
}

async function initWatchlist(){
  const c=await fetch('/api/config').then(r=>r.json()).catch(()=>({watchlist:[]}));
  document.getElementById('wl-input').value=(c.watchlist||[]).join(', ');
}

function setUpdated(){
  const t=new Date();
  document.getElementById('upd-time').textContent=
    t.toLocaleTimeString('en-US',{hour:'2-digit',minute:'2-digit',second:'2-digit'});
}
async function refreshAll(){
  const btn=document.getElementById('refresh-btn');
  btn.textContent='Refreshing…'; btn.disabled=true;
  await fetch('/api/refresh');
  await Promise.all([loadIndices(),loadSectors(),loadAssets()]);
  scData=[]; document.getElementById('scrn').innerHTML='<tr><td colspan="11" class="loading">Refreshing…</td></tr>';
  await loadScreener();
  btn.textContent='✓ Refreshed'; setTimeout(()=>{btn.textContent='⟳ Refresh All';btn.disabled=false;},2000);
  setUpdated();
}

loadIndices(); loadSectors(); loadAssets(); loadScreener(); initWatchlist();
setUpdated();
</script>
</body></html>
"""

# ── News page ──────────────────────────────────────────────────────────────────
NEWS_HTML = """<!DOCTYPE html><html lang="en"><head>
<meta charset="UTF-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>World News — DailyPulse</title>""" + CSS + """</head><body>
""" + NAV.replace('{MA}','').replace('{PA}','').replace('{LA}','').replace('{CA}','').replace('{NA}','active').replace('{AA}','').replace('{DA}','') + """
<div class="wrap">
<div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:8px;flex-wrap:wrap;gap:8px;">
  <div style="font-size:12px;color:var(--muted)">Updated: <span id="news-upd" style="color:var(--acc)">loading…</span></div>
  <button class="fbtn" onclick="refreshNews()" id="news-refresh-btn" style="font-size:11px;padding:4px 12px">&#8635; Refresh News</button>
</div>
<div class="cat-tabs">
  <div class="ctab on" data-c="all">All</div>
  <div class="ctab" data-c="Global">Global</div>
  <div class="ctab" data-c="Geopolitics">Geopolitics</div>
  <div class="ctab" data-c="Policy">Policy</div>
  <div class="ctab" data-c="Nat. Security">Nat. Security</div>
  <div class="ctab" data-c="Economy">Economy</div>
</div>
<div class="news-grid" id="ng"><div class="loading" style="grid-column:1/-1">Loading world news…</div></div>
</div>
<script>
function timeAgo(s){
  if(!s)return'';
  try{const d=new Date(s),diff=(Date.now()-d)/1000;
    return diff<3600?Math.round(diff/60)+'m ago':
           diff<86400?Math.round(diff/3600)+'h ago':
           Math.round(diff/86400)+'d ago';}catch{return'';}
}
function catCls(c){
  return c==='Global'?'cat-Global':c==='Geopolitics'?'cat-Geopolitics':
         c==='Policy'?'cat-Policy':c==='Nat. Security'?'cat-NatSecurity':'cat-Economy';
}
async function loadNews(cat){
  document.querySelectorAll('.ctab').forEach(t=>t.classList.toggle('on',t.dataset.c===cat));
  const ng=document.getElementById('ng');
  ng.innerHTML='<div class="loading" style="grid-column:1/-1">Loading…</div>';
  const url=cat==='all'?'/api/news':`/api/news?category=${encodeURIComponent(cat)}`;
  const data=await fetch(url).then(r=>r.json()).catch(()=>null);
  if(!data||!data.length){ng.innerHTML='<div class="loading" style="grid-column:1/-1">No articles found.</div>';return;}
  ng.innerHTML=data.map(a=>`
    <div class="nc">
      <div class="nc-top">
        <span class="ns-badge">${a.src}</span>
        <span class="nc-cat ${catCls(a.category)}">${a.category}</span>
        <span class="nc-time">${timeAgo(a.published)}</span>
      </div>
      <div class="nc-title"><a href="${a.link}" target="_blank" rel="noopener">${a.title}</a></div>
      ${a.summary?`<div class="nc-summ">${a.summary}</div>`:''}
    </div>`).join('');
}
function setNewsUpdated(){
  const t=new Date();
  document.getElementById('news-upd').textContent=
    t.toLocaleTimeString('en-US',{hour:'2-digit',minute:'2-digit',second:'2-digit'});
}
async function refreshNews(){
  const btn=document.getElementById('news-refresh-btn');
  btn.textContent='Refreshing…'; btn.disabled=true;
  await fetch('/api/refresh');
  await loadNews(currentCat);
  btn.textContent='✓ Refreshed'; setTimeout(()=>{btn.textContent='⟳ Refresh News';btn.disabled=false;},2000);
  setNewsUpdated();
}
document.querySelectorAll('.ctab').forEach(t=>t.addEventListener('click',()=>loadNews(t.dataset.c)));
loadNews('all'); setNewsUpdated();
</script>
</body></html>
"""

# ── Ask page ───────────────────────────────────────────────────────────────────
CALENDAR_HTML = """<!DOCTYPE html><html lang="en"><head>
<meta charset="UTF-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Calendar — DailyPulse</title>""" + CSS + """</head><body>
""" + NAV.replace('{MA}','').replace('{PA}','').replace('{LA}','').replace('{CA}','active').replace('{NA}','').replace('{AA}','').replace('{DA}','') + """
<div class="wrap">

<div style="display:grid;grid-template-columns:1fr 1fr;gap:24px" class="cal-grid">

<div class="sec">
  <div class="sec-title">Economic Calendar</div>
  <div id="econ-list"><div class="loading">Loading…</div></div>
</div>

<div class="sec">
  <div class="sec-title">Upcoming Earnings</div>
  <div id="earn-list"><div class="loading">Fetching earnings dates (30–60 sec)…</div></div>
</div>

</div>
</div>

<style>
@media(max-width:900px){.cal-grid{grid-template-columns:1fr!important;}}
</style>

<script>
const TYPE_LABELS={'fomc':'FOMC','cpi':'CPI','nfp':'NFP','pce':'PCE',
  'gdp':'GDP','retail':'Retail','ism':'ISM','earnings':'Earnings'};

function fmtDate(ds){
  const d=new Date(ds+'T12:00:00');
  const today=new Date(); today.setHours(0,0,0,0);
  const dt=new Date(ds+'T00:00:00');
  const diff=Math.round((dt-today)/86400000);
  const label=diff===0?'Today':diff===1?'Tomorrow':diff===-1?'Yesterday':
    d.toLocaleDateString('en-US',{month:'short',day:'numeric'});
  const dow=d.toLocaleDateString('en-US',{weekday:'short'});
  const isToday=diff===0; const isSoon=diff>=0&&diff<=3;
  return{label,dow,isToday,isSoon,diff};
}

async function loadEconomic(){
  const data=await fetch('/api/calendar/economic').then(r=>r.json()).catch(()=>[]);
  const el=document.getElementById('econ-list');
  if(!data.length){el.innerHTML='<div class="loading">No upcoming events</div>';return;}
  const typeCls={'fomc':'type-fomc','cpi':'type-cpi','pce':'type-pce','nfp':'type-nfp',
    'gdp':'type-gdp','retail':'type-retail','ism':'type-ism'};
  el.innerHTML=data.map(e=>{
    const {label,dow,isToday,isSoon}=fmtDate(e.date);
    const cls=isToday?'cal-today':isSoon?'cal-upcoming':'';
    return`<div class="cal-event ${cls}">
      <div class="cal-date-col"><div class="cal-date">${label}</div><div class="cal-dow">${dow}</div></div>
      <div style="flex:1">
        <div style="display:flex;align-items:center;gap:8px;margin-bottom:4px">
          <span class="cal-type ${typeCls[e.type]||'type-ism'}">${TYPE_LABELS[e.type]||e.type}</span>
          <span class="cal-name">${e.event}</span>
          ${e.importance==='high'?'<span style="color:var(--neg);font-size:11px;font-weight:700">HIGH</span>':''}
        </div>
        <div class="cal-note">${e.note}</div>
      </div>
    </div>`;
  }).join('');
}

async function loadEarnings(){
  const data=await fetch('/api/calendar/earnings').then(r=>r.json()).catch(()=>[]);
  const el=document.getElementById('earn-list');
  if(!data.length){el.innerHTML='<div class="loading">No earnings found for watchlist</div>';return;}
  el.innerHTML=data.map(e=>{
    const {label,dow,isToday,isSoon}=fmtDate(e.date);
    const cls=isToday?'cal-today':isSoon?'cal-upcoming':'';
    return`<div class="cal-event ${cls}">
      <div class="cal-date-col"><div class="cal-date">${label}</div><div class="cal-dow">${dow}</div></div>
      <div style="flex:1">
        <div style="display:flex;align-items:center;gap:8px">
          <span class="cal-type type-earnings">EARN</span>
          <span class="cal-name" style="font-family:'SF Mono',monospace;font-weight:700">${e.symbol}</span>
        </div>
      </div>
    </div>`;
  }).join('');
}

loadEconomic();
loadEarnings();
</script>
</body></html>
"""

# ── Digest page ────────────────────────────────────────────────────────────────
@app.route('/digest')
def digest_page(): return render_template_string(DIGEST_HTML)

DIGEST_HTML = """<!DOCTYPE html><html lang="en"><head>
<meta charset="UTF-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Digest — DailyPulse</title>""" + CSS + """</head><body>
""" + NAV.replace('{MA}','').replace('{LA}','active').replace('{CA}','').replace('{NA}','') + """
<div class="wrap">
  <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:12px;flex-wrap:wrap;gap:8px;">
    <div>
      <h1 style="font-size:20px;margin:0">Daily Digest</h1>
      <div style="font-size:12px;color:var(--muted)" id="dg-date">—</div>
    </div>
    <div style="display:flex;align-items:center;gap:10px">
      <span style="font-size:11px;color:var(--muted)">Updated <span id="dg-upd" style="color:var(--acc)">…</span></span>
      <button class="fbtn" onclick="loadDigest()" id="dg-btn" style="font-size:11px;padding:4px 12px">&#8635; Refresh</button>
    </div>
  </div>
  <div id="dg-wrap"><div class="loading">Building digest…</div></div>
</div>
<script>
function pc(v){const s=v>=0?'+':'';return `<span style="color:${v>=0?'var(--pos)':'var(--neg)'}">${s}${v.toFixed(2)}%</span>`;}
function esc(s){return (s||'').replace(/[&<>]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;'}[c]));}
function card(title,inner){return `<div class="sec" style="margin-bottom:18px"><div class="sec-title">${title}</div>${inner}</div>`;}
async function loadDigest(){
  const wrap=document.getElementById('dg-wrap'), btn=document.getElementById('dg-btn');
  wrap.innerHTML='<div class="loading">Building digest…</div>'; btn.disabled=true;
  let d;
  try{ d=await fetch('/api/digest').then(r=>r.json()); }
  catch(e){ wrap.innerHTML='<div class="loading" style="color:var(--neg)">Could not load digest.</div>'; btn.disabled=false; return; }
  render(d); btn.disabled=false;
}
function render(d){
  document.getElementById('dg-date').textContent=d.date||'';
  document.getElementById('dg-upd').textContent=d.generated_at||'';
  const b=d.breadth||{up:0,down:0,total:0};
  const tone=d.spx==null?'Mixed':(d.spx>0.3?'Risk-on':(d.spx<-0.3?'Risk-off':'Flat'));
  const chip=(t,v)=>`<span style="background:var(--bg3);border:1px solid var(--border);border-radius:6px;padding:4px 10px;font-size:12px">${esc(t)} ${pc(v)}</span>`;
  let html='';
  html+=card('Market Pulse',
    `<div style="display:flex;flex-wrap:wrap;gap:18px;margin-bottom:12px">
       <div><div style="font-size:11px;color:var(--muted)">Tone</div><div style="font-size:16px;font-weight:600">${tone}</div></div>
       <div><div style="font-size:11px;color:var(--muted)">Sector breadth</div><div style="font-size:16px;font-weight:600"><span style="color:var(--pos)">${b.up}↑</span> / <span style="color:var(--neg)">${b.down}↓</span> of ${b.total}</div></div>
       <div><div style="font-size:11px;color:var(--muted)">Headlines scanned</div><div style="font-size:16px;font-weight:600">${d.news_total||0}</div></div>
     </div>
     <div style="display:flex;flex-wrap:wrap;gap:8px">${(d.indices||[]).map(i=>chip(i.name,i.pct)).join('')}</div>`);
  html+=card('Biggest Movers',`<div style="display:flex;flex-wrap:wrap;gap:8px">${(d.movers||[]).map(m=>chip(m.name,m.pct)).join('')}</div>`);
  const rows=(arr)=>arr.map(s=>`<div style="display:flex;justify-content:space-between;padding:3px 0;font-size:13px"><span>${esc(s.name)}</span>${pc(s.pct)}</div>`).join('');
  html+=`<div style="display:flex;gap:18px;flex-wrap:wrap"><div style="flex:1;min-width:220px">${card('Sector Leaders',rows(d.sectors_top||[]))}</div><div style="flex:1;min-width:220px">${card('Sector Laggards',rows(d.sectors_bottom||[]))}</div></div>`;
  html+=card('Top Headlines',(d.headlines||[]).map(h=>`<div style="padding:6px 0;border-bottom:1px solid var(--border)"><a href="${esc(h.link)}" target="_blank" rel="noopener" style="color:var(--text);text-decoration:none;font-size:13px">${esc(h.title)}</a><div style="font-size:11px;color:var(--muted)">${esc(h.category)} · ${esc(h.src)}</div></div>`).join(''));
  html+=card('By Category',(d.news_by_cat||[]).map(c=>`<div style="margin-bottom:10px"><div style="font-size:12px;font-weight:600;color:var(--acc)">${esc(c.category)} <span style="color:var(--muted);font-weight:400">(${c.count})</span></div>${(c.top||[]).map(t=>`<div style="font-size:12px;padding:2px 0"><a href="${esc(t.link)}" target="_blank" rel="noopener" style="color:var(--muted);text-decoration:none">• ${esc(t.title)}</a></div>`).join('')}</div>`).join(''));
  document.getElementById('dg-wrap').innerHTML=html;
}
loadDigest();
</script>
</body></html>
"""


# ── entry point ────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    port = int(os.environ.get('PORT', cfg().get('port', 5055)))
    print(f"\n  DailyPulse (public)  →  http://0.0.0.0:{port}")
    print("  Ctrl+C to stop\n")
    app.run(host='0.0.0.0', port=port, debug=False, use_reloader=False)
