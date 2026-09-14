# функции для анализа и запросы к api, чтобы не дублировать это в ноутбуке и в приложении
import asyncio
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor

import aiohttp
import numpy as np
import pandas as pd
import requests

OWM_URL = "https://api.openweathermap.org/data/2.5/weather"

MONTH_TO_SEASON = {12: "winter", 1: "winter", 2: "winter",
                   3: "spring", 4: "spring", 5: "spring",
                   6: "summer", 7: "summer", 8: "summer",
                   9: "autumn", 10: "autumn", 11: "autumn"}

SEASON_RU = {"winter": "зима", "spring": "весна", "summer": "лето", "autumn": "осень"}
SEASON_ORDER = ["winter", "spring", "summer", "autumn"]

WINDOW = 30  # окно скользящего среднего в днях


def load_data(src):
    df = pd.read_csv(src, parse_dates=["timestamp"])
    return df.sort_values(["city", "timestamp"]).reset_index(drop=True)


def analyze_city(df_city):
    # анализ одного города, вынесен отдельно чтобы можно было закинуть в пул процессов
    d = df_city.sort_values("timestamp").copy()

    # скольящее окно
    d["roll_mean"] = d["temperature"].rolling(WINDOW, min_periods=1).mean()
    d["roll_std"] = d["temperature"].rolling(WINDOW, min_periods=1).std()
    d["anomaly_roll"] = (d["temperature"] - d["roll_mean"]).abs() > 2 * d["roll_std"]

    # сезон
    g = d.groupby("season")["temperature"]
    d["season_mean"] = g.transform("mean")
    d["season_std"] = g.transform("std")
    d["anomaly_season"] = (d["temperature"] - d["season_mean"]).abs() > 2 * d["season_std"]

    d["anomaly_roll"] = d["anomaly_roll"].fillna(False)
    return d


def analyze_sequential(df):
    return pd.concat([analyze_city(g) for _, g in df.groupby("city")], ignore_index=True)


def analyze_parallel(df, n_workers=4, kind="process"):
    # то же самое, но города раскиданы по процессам или потокам
    parts = [g for _, g in df.groupby("city")]
    pool = ProcessPoolExecutor if kind == "process" else ThreadPoolExecutor
    with pool(max_workers=n_workers) as ex:
        res = list(ex.map(analyze_city, parts))
    return pd.concat(res, ignore_index=True)


def analyze_vectorized(df):
    d = df.sort_values(["city", "timestamp"]).copy()
    g = d.groupby("city")["temperature"]
    d["roll_mean"] = g.transform(lambda s: s.rolling(WINDOW, min_periods=1).mean())
    d["roll_std"] = g.transform(lambda s: s.rolling(WINDOW, min_periods=1).std())
    d["anomaly_roll"] = ((d["temperature"] - d["roll_mean"]).abs() > 2 * d["roll_std"]).fillna(False)

    gs = d.groupby(["city", "season"])["temperature"]
    d["season_mean"] = gs.transform("mean")
    d["season_std"] = gs.transform("std")
    d["anomaly_season"] = (d["temperature"] - d["season_mean"]).abs() > 2 * d["season_std"]
    return d.reset_index(drop=True)


def season_profile(d):
    p = (d.groupby("season")["temperature"]
           .agg(mean="mean", std="std", min="min", max="max", n="count")
           .reindex(SEASON_ORDER)
           .reset_index())
    p["low"] = p["mean"] - 2 * p["std"]
    p["high"] = p["mean"] + 2 * p["std"]
    p["сезон"] = p["season"].map(SEASON_RU)
    return p


def trend(d):
    # линейная регрессия по времени
    x = d["timestamp"].map(pd.Timestamp.toordinal).to_numpy(dtype=float)
    y = d["temperature"].to_numpy(dtype=float)
    k, b = np.polyfit(x, y, 1)
    return {"slope_per_year": k * 365.25, "line": k * x + b}


def city_summary(d):
    t = trend(d)
    return {
        "city": d["city"].iloc[0],
        "mean": d["temperature"].mean(),
        "min": d["temperature"].min(),
        "max": d["temperature"].max(),
        "std": d["temperature"].std(),
        "anomalies": int(d["anomaly_season"].sum()),
        "anomalies_%": 100 * d["anomaly_season"].mean(),
        "trend_C_per_year": t["slope_per_year"],
    }


def current_season(ts=None):
    ts = ts or pd.Timestamp.now()
    return MONTH_TO_SEASON[ts.month]


def is_normal(temp, d, season):
    # сравниваем текущую температуру с нормой сезона
    s = d[d["season"] == season]["temperature"]
    mean, std = s.mean(), s.std()
    low, high = mean - 2 * std, mean + 2 * std
    return {
        "season": season, "mean": mean, "std": std, "low": low, "high": high,
        "normal": bool(low <= temp <= high),
        "dev": temp - mean,
    }


def get_weather_sync(city, api_key):
    # возвращаю (температура, ошибка), при кривом ключе в ошибке лежит json от api
    try:
        r = requests.get(OWM_URL, params={"q": city, "appid": api_key, "units": "metric"}, timeout=10)
        js = r.json()
    except Exception as e:
        return None, {"cod": "error", "message": str(e)}
    if r.status_code != 200:
        return None, js
    return js["main"]["temp"], None


def get_weather_sync_many(cities, api_key):
    # города по очереди, одной сессией
    out = {}
    with requests.Session() as s:
        for c in cities:
            try:
                r = s.get(OWM_URL, params={"q": c, "appid": api_key, "units": "metric"}, timeout=10)
                js = r.json()
                out[c] = (js["main"]["temp"], None) if r.status_code == 200 else (None, js)
            except Exception as e:
                out[c] = (None, {"cod": "error", "message": str(e)})
    return out


async def _get_one(session, city, api_key):
    try:
        async with session.get(OWM_URL, params={"q": city, "appid": api_key, "units": "metric"}) as r:
            js = await r.json()
            return city, ((js["main"]["temp"], None) if r.status == 200 else (None, js))
    except Exception as e:
        return city, (None, {"cod": "error", "message": str(e)})


async def get_weather_async_many(cities, api_key):
    # все города разом
    timeout = aiohttp.ClientTimeout(total=15)
    async with aiohttp.ClientSession(timeout=timeout) as s:
        res = await asyncio.gather(*[_get_one(s, c, api_key) for c in cities])
    return dict(res)


def _run(coro):
    # в jupyter уже крутится свой event loop и asyncio.run падает, поэтому такой костыль
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)
    with ThreadPoolExecutor(1) as ex:
        return ex.submit(asyncio.run, coro).result()


def get_weather_async(city, api_key):
    return _run(get_weather_async_many([city], api_key))[city]


def get_weather_async_many_blocking(cities, api_key):
    return _run(get_weather_async_many(cities, api_key))


def get_weather_threads_many(cities, api_key, n_workers=8):
    # третий вариант, обычные запросы но в потоках
    with ThreadPoolExecutor(max_workers=n_workers) as ex:
        res = list(ex.map(lambda c: (c, get_weather_sync(c, api_key)), cities))
    return dict(res)
