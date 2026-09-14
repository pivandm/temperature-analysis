# дз 1, анализ температур и текущая погода с openweathermap
# запуск: streamlit run app.py
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

import utils

st.set_page_config(page_title="Анализ температур", layout="wide")

DEFAULT_CSV = "temperature_data.csv"

@st.cache_data(show_spinner="Считаю...")
def prepare(src):
    # кешируем, иначе пересчитывается на каждый чих в интерфейсе
    df = utils.load_data(src)
    return utils.analyze_vectorized(df)

st.sidebar.title("Настройки")

up = st.sidebar.file_uploader("Файл с историческими данными (csv)", type="csv")
data = prepare(up if up is not None else DEFAULT_CSV)  # без загрузки берем файл из репозитория

cities = sorted(data["city"].unique())
city = st.sidebar.selectbox("Город", cities, index=cities.index("Moscow") if "Moscow" in cities else 0)

anom_mode = st.sidebar.radio(
    "Как считаем аномалии",
    ["По сезонной норме (mean +- 2std)", "По скользящему окну (30 дней +- 2std)"],
)
anom_col = "anomaly_season" if anom_mode.startswith("По сезонной") else "anomaly_roll"

st.sidebar.divider()
api_key = st.sidebar.text_input("API-ключ OpenWeatherMap", type="password",
                                help="Без ключа текущая погода не показывается")
how = st.sidebar.radio("Как обращаться к API", ["синхронно (requests)", "асинхронно (aiohttp)"])

d = data[data["city"] == city].reset_index(drop=True)
prof = utils.season_profile(d)
tr = utils.trend(d)

st.title(f"Анализ температуры: {city}")

tab1, tab2, tab3 = st.tabs(["Исторические данные", "Текущая погода", "Все города"])

# история
with tab1:
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Средняя", f"{d['temperature'].mean():.1f} C")
    c2.metric("Минимум", f"{d['temperature'].min():.1f} C")
    c3.metric("Максимум", f"{d['temperature'].max():.1f} C")
    c4.metric("Аномалий", f"{int(d[anom_col].sum())}", f"{100 * d[anom_col].mean():.1f}% дней")

    slope = tr["slope_per_year"]
    st.info(f"**Тренд:** {slope:+.3f} C в год "
            f"({'теплеет' if slope > 0 else 'холодает'}, за 10 лет {slope * 10:+.1f} C). "
            "Данные синтетические, так что около нуля это нормально.")

    st.subheader("Описательная статистика")
    left, right = st.columns([1, 2])
    with left:
        st.write("По всем данным:")
        st.dataframe(d["temperature"].describe().to_frame("temperature"))
    with right:
        st.write("По сезонам:")
        st.dataframe(
            prof[["сезон", "mean", "std", "min", "max", "n"]]
            .rename(columns={"mean": "средняя", "min": "мин", "max": "макс", "n": "дней"}),
            hide_index=True,
        )

    st.subheader("Временной ряд и аномалии")
    an = d[d[anom_col]]
    if anom_col == "anomaly_roll":
        lo, hi = d["roll_mean"] - 2 * d["roll_std"], d["roll_mean"] + 2 * d["roll_std"]
    else:
        lo, hi = d["season_mean"] - 2 * d["season_std"], d["season_mean"] + 2 * d["season_std"]

    fig = go.Figure()
    # две линии и заливка между ними, это коридор нормы
    fig.add_trace(go.Scatter(x=d["timestamp"], y=hi, line=dict(width=0), showlegend=False,
                             hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=d["timestamp"], y=lo, line=dict(width=0), fill="tonexty",
                             fillcolor="rgba(100,150,255,0.15)", name="норма +-2std", hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=d["timestamp"], y=d["temperature"], name="температура",
                             line=dict(color="lightgray", width=1)))
    fig.add_trace(go.Scatter(x=d["timestamp"], y=d["roll_mean"], name="скользящее среднее (30д)",
                             line=dict(color="royalblue", width=2)))
    fig.add_trace(go.Scatter(x=d["timestamp"], y=tr["line"], name="тренд",
                             line=dict(color="green", width=2, dash="dash")))
    fig.add_trace(go.Scatter(x=an["timestamp"], y=an["temperature"], mode="markers", name="аномалии",
                             marker=dict(color="red", size=5)))
    fig.update_layout(height=500, hovermode="x unified", yaxis_title="C",
                      legend=dict(orientation="h", y=1.02, yanchor="bottom"))
    fig.update_xaxes(rangeslider_visible=True)
    st.plotly_chart(fig)

    st.subheader("Сезонный профиль")
    cc1, cc2 = st.columns(2)
    with cc1:
        f2 = px.bar(prof, x="сезон", y="mean", error_y="std", color="сезон",
                    labels={"mean": "средняя температура, C"},
                    title="Средняя и std по сезонам")
        f2.update_layout(showlegend=False, height=400)
        st.plotly_chart(f2)
    with cc2:
        f3 = px.box(d, x="season", y="temperature", color="season",
                    category_orders={"season": utils.SEASON_ORDER},
                    title="Распределение температур по сезонам",
                    labels={"temperature": "C", "season": "сезон"})
        f3.update_layout(showlegend=False, height=400)
        st.plotly_chart(f3)

    st.subheader("Средняя температура по годам и месяцам")
    hm = (d.assign(year=d["timestamp"].dt.year, month=d["timestamp"].dt.month)
            .pivot_table(index="year", columns="month", values="temperature", aggfunc="mean"))
    f4 = px.imshow(hm, color_continuous_scale="RdBu_r", aspect="auto",
                   labels=dict(x="месяц", y="год", color="C"), text_auto=".1f")
    f4.update_layout(height=400)
    st.plotly_chart(f4)

    with st.expander("Таблица аномальных дней"):
        st.dataframe(an[["timestamp", "temperature", "season", "season_mean", "roll_mean"]],
                     hide_index=True)

with tab2:
    st.subheader("Текущая температура через OpenWeatherMap")

    if not api_key:
        st.info("Введите API-ключ в сайдбаре, чтобы увидеть текущую погоду.")
    else:
        if how.startswith("син"):
            temp, err = utils.get_weather_sync(city, api_key)
        else:
            temp, err = utils.get_weather_async(city, api_key)

        if err is not None:
            st.error(f"API вернул ошибку: {err}")
        else:
            season = utils.current_season()
            chk = utils.is_normal(temp, d, season)

            m1, m2, m3 = st.columns(3)
            m1.metric("Сейчас", f"{temp:.1f} C", f"{chk['dev']:+.1f} C к норме сезона")
            m2.metric(f"Норма ({utils.SEASON_RU[season]})", f"{chk['mean']:.1f} C", f"std = {chk['std']:.1f}")
            m3.metric("Диапазон нормы", f"{chk['low']:.1f} ... {chk['high']:.1f} C")

            if chk["normal"]:
                st.success(f"Температура **нормальная** для сезона {utils.SEASON_RU[season]}: "
                           f"попадает в {chk['low']:.1f}...{chk['high']:.1f} C.")
            else:
                st.error(f"Температура **аномальная**! Выходит за диапазон "
                         f"{chk['low']:.1f}...{chk['high']:.1f} C для сезона {utils.SEASON_RU[season]}.")

            hist = d[d["season"] == season]["temperature"]
            f5 = px.histogram(hist, nbins=40, title=f"Где мы на историческом распределении ({utils.SEASON_RU[season]})",
                              labels={"value": "C"})
            f5.add_vline(x=chk["low"], line_dash="dot", line_color="orange")
            f5.add_vline(x=chk["high"], line_dash="dot", line_color="orange")
            f5.add_vline(x=temp, line_color="red", line_width=3,
                         annotation_text="сейчас", annotation_position="top")
            f5.update_layout(showlegend=False, height=400)
            st.plotly_chart(f5)

    st.caption("Для одного города sync и async работают одинаково. Async выигрывает когда городов много, "
               "это видно во вкладке Все города.")

with tab3:
    st.subheader("Сводка по всем городам")
    summary = pd.DataFrame([utils.city_summary(g) for _, g in data.groupby("city")])
    st.dataframe(
        summary.rename(columns={"city": "город", "mean": "средняя",
                                "anomalies": "аномалий", "anomalies_%": "% аномалий",
                                "trend_C_per_year": "тренд, C/год"}).round(2),
        hide_index=True,
    )

    f6 = px.bar(summary.sort_values("mean"), x="city", y="mean", error_y="std",
                title="Средняя температура по городам", labels={"mean": "C", "city": "город"})
    st.plotly_chart(f6)

    st.divider()
    st.write("**Текущая погода во всех городах сразу**")
    if not api_key:
        st.info("Нужен API-ключ.")
    elif st.button("Запросить все города"):
        import time
        t0 = time.perf_counter()
        res = (utils.get_weather_sync_many(cities, api_key) if how.startswith("син")
               else utils.get_weather_async_many_blocking(cities, api_key))
        dt = time.perf_counter() - t0

        rows = []
        for c, (t, e) in res.items():
            if e is not None:
                rows.append({"город": c, "сейчас, C": None, "статус": f"ошибка: {e.get('message', e)}"})
                continue
            dc = data[data["city"] == c]
            chk = utils.is_normal(t, dc, utils.current_season())
            rows.append({"город": c, "сейчас, C": round(t, 1),
                         "норма сезона, C": round(chk["mean"], 1),
                         "диапазон": f"{chk['low']:.1f}...{chk['high']:.1f}",
                         "статус": "норма" if chk["normal"] else "АНОМАЛИЯ"})
        st.caption(f"{len(cities)} запросов за {dt:.2f} сек ({how})")
        st.dataframe(pd.DataFrame(rows), hide_index=True)
