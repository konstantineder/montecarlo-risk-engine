const chartHosts = new Map();
let chartEngine;
let chartModal;

const chartColors = {
  accent: "#6251b5",
  gain: "#a36916",
  loss: "#b84048",
  grid: "#e9e8ef",
  label: "#626779",
};

const compactCurrency = new Intl.NumberFormat("en-US", {
  style: "currency", currency: "USD", notation: "compact", maximumFractionDigits: 1,
});
const dollars = new Intl.NumberFormat("en-US", {
  style: "currency", currency: "USD", maximumFractionDigits: 2,
});

function loadChartEngine() {
  chartEngine ||= import("/charts/engine.js").catch((error) => {
    chartEngine = undefined;
    throw error;
  });
  return chartEngine;
}

function mountChart(host, spec) {
  const entry = {};
  chartHosts.set(host, entry);
  const render = async () => {
    entry.observer?.disconnect();
    try {
      const engine = await loadChartEngine();
      if (chartHosts.get(host) !== entry || !host.isConnected) return;
      const chart = engine.init(host, null, { renderer: "canvas", devicePixelRatio: Math.min(devicePixelRatio || 1, 2) });
      entry.chart = chart;
      chart.setOption(spec.options());
      if (spec.click) chart.on("click", spec.click);
      if (spec.selectDay) chart.getZr().on("click", (event) => {
        const point = [event.offsetX, event.offsetY];
        if (!chart.containPixel({ gridIndex: 0 }, point)) return;
        const [day] = chart.convertFromPixel({ gridIndex: 0 }, point);
        spec.selectDay(Math.round(day));
        chartModal?.close();
      });
      if (spec.cursor) {
        const update = () => chart.setOption({ series: [{ id: spec.cursor.id, markLine: markers(spec.cursor.zero, spec.cursor.day()) }] });
        host.addEventListener("desk:day", update);
        update();
      }
      if (typeof ResizeObserver !== "undefined") {
        entry.resize = new ResizeObserver(() => chart.resize());
        entry.resize.observe(host);
      }
    } catch {
      if (chartHosts.get(host) === entry) host.textContent = "Chart unavailable. Use the data table below.";
    }
  };
  if (typeof IntersectionObserver !== "undefined") {
    entry.observer = new IntersectionObserver((entries) => {
      if (entries.some((item) => item.isIntersecting)) render();
    }, { rootMargin: "160px" });
    entry.observer.observe(host);
  } else if (typeof ResizeObserver !== "undefined") {
    queueMicrotask(render);
  }
}

function disposeChart(host) {
  const entry = chartHosts.get(host);
  if (!entry) return;
  entry.observer?.disconnect();
  entry.resize?.disconnect();
  entry.chart?.dispose();
  chartHosts.delete(host);
}

function disposeCharts() {
  chartModal?.close();
  for (const host of chartHosts.keys()) disposeChart(host);
}

function chartCard(title, note, spec) {
  const card = document.createElement("article");
  card.className = "chart-card";
  const head = document.createElement("header");
  const copy = document.createElement("div");
  copy.innerHTML = `<h3>${title}</h3><p>${note}</p>`;
  const button = document.createElement("button");
  button.className = "max";
  button.textContent = "Maximize";
  button.setAttribute("aria-label", `Maximize ${title}`);
  button.addEventListener("click", () => {
    const modal = document.createElement("dialog");
    modal.className = "chart-dialog";
    modal.setAttribute("aria-label", title);
    modal.innerHTML = `<header><div><h2>${title}</h2><p class="note">${note}</p></div><button class="ghost">Close</button></header>`;
    const body = chartSurface(title);
    modal.append(body);
    modal.querySelector("button").addEventListener("click", () => modal.close());
    modal.addEventListener("close", () => {
      disposeChart(body);
      modal.remove();
      if (chartModal === modal) chartModal = undefined;
      if (button.isConnected) button.focus();
    }, { once: true });
    document.body.append(modal);
    chartModal = modal;
    modal.showModal();
    mountChart(body, spec);
  });
  const body = chartSurface(title);
  head.append(copy, button);
  card.append(head, body);
  mountChart(body, spec);
  return card;
}

function chartSurface(title) {
  const body = document.createElement("div");
  body.className = "chart-surface";
  body.setAttribute("role", "img");
  body.setAttribute("aria-label", title);
  return body;
}

function axis(name, formatter, extra = {}) {
  return {
    type: "value", name, nameLocation: "middle", nameGap: 27,
    nameTextStyle: { color: chartColors.label },
    axisLabel: { color: chartColors.label, formatter, hideOverlap: true },
    axisLine: { show: false },
    axisTick: { show: false },
    splitLine: { lineStyle: { color: chartColors.grid } },
    ...extra,
  };
}

function chartOptions(xAxis, yAxis, extra = {}) {
  return {
    animation: false,
    textStyle: { fontFamily: "system-ui, sans-serif" },
    grid: { left: 68, right: 20, top: 35, bottom: 65 },
    tooltip: { confine: true, renderMode: "richText" },
    xAxis, yAxis,
    ...extra,
  };
}

function zoom() {
  return [
    { type: "inside", zoomOnMouseWheel: "ctrl", moveOnMouseWheel: false },
    { type: "slider", bottom: 7, height: 16, showDetail: false, showDataShadow: false, brushSelect: false },
  ];
}

function markers(zero, day) {
  return {
    silent: true, symbol: "none", label: { show: false },
    lineStyle: { type: "dashed", color: "#80869a" },
    data: [...(zero ? [{ yAxis: 0 }] : []), ...(day == null ? [] : [{ xAxis: day, lineStyle: { color: chartColors.accent } }])],
  };
}

function multiChart(rows, series, opts = {}) {
  const valueLabel = opts.money ? (value) => compactCurrency.format(value) : undefined;
  const exact = opts.money ? (value) => dollars.format(value) : (value) => Number(value).toLocaleString("en-US", { maximumFractionDigits: 2 });
  const cursor = opts.day ? { id: series[0].key, day: opts.day, zero: opts.zero } : undefined;
  // Stack percentiles along the value axis by making trading days categories.
  const dayAxis = opts.band
    ? axis("Trading day", undefined, { type: "category", boundaryGap: false, data: rows.map((row) => String(row.day)) })
    : axis("Trading day", undefined, { minInterval: 1 });
  return {
    cursor,
    selectDay: opts.selectDay,
    options: () => chartOptions(dayAxis, axis("", valueLabel, { scale: !opts.zero }), {
      legend: { type: "scroll", top: 0, data: series.filter((item) => !item.hidden).map((item) => item.label) },
      tooltip: {
        trigger: "axis", confine: true, renderMode: "richText",
        formatter: (points) => {
          const day = points[0]?.value[0];
          if (day == null) return "";
          const lines = points.filter((point) => !series[point.seriesIndex].hidden)
            .map((point) => `${point.marker} ${point.seriesName}  ${exact(point.value[1])}`);
          if (opts.band) {
            const row = rows[points[0].dataIndex];
            lines.push(`5th–95th  ${exact(row[opts.band[0]])} to ${exact(row[opts.band[1]])}`);
          }
          return [`Day ${day}`, ...lines].join("\n");
        },
      },
      dataZoom: zoom(),
      series: series.map((item, index) => ({
        id: item.key, type: item.type || "line", name: item.label, showSymbol: false,
        symbolSize: item.type === "scatter" ? 7 : 5,
        color: item.color, lineStyle: { width: item.hidden ? 0 : 2.25 },
        silent: !!item.hidden,
        stack: item.stack, stackStrategy: item.stack ? "all" : undefined,
        areaStyle: item.areaStyle, step: item.step,
        markLine: index === 0 && (opts.zero || opts.day) ? markers(opts.zero, opts.day?.()) : undefined,
        data: rows.map((row) => [opts.band ? String(row.day) : row.day, row[item.key] ?? null]),
      })),
    }),
  };
}

function bandChart(rows, low, mid, high, opts = {}) {
  const data = rows.map((row) => ({ ...row, bandWidth: row[high] - row[low] }));
  return multiChart(data, [
    { key: low, label: "Lower bound", hidden: true, stack: "band", color: chartColors.accent, areaStyle: { opacity: 0 } },
    { key: "bandWidth", label: "Percentile range", hidden: true, stack: "band", color: chartColors.accent, areaStyle: { opacity: 0.17 } },
    { key: mid, label: "Median", color: chartColors.accent },
  ], { ...opts, band: [low, high] });
}

function scatter(rows, openPath, xKey = "ending_spot", xLabel = "Ending spot") {
  // Quantiles and the table use every path. Cap the interactive point layer.
  const stride = Math.ceil(rows.length / 2500);
  const sample = rows.filter((_, index) => index % stride === 0);
  if (stride > 1) {
    const extremes = rows.reduce((acc, row) => [
      row[xKey] < acc[0][xKey] ? row : acc[0],
      row[xKey] > acc[1][xKey] ? row : acc[1],
      row.pnl < acc[2].pnl ? row : acc[2],
      row.pnl > acc[3].pnl ? row : acc[3],
    ], Array(4).fill(rows[0]));
    sample.push(...[...new Set(extremes)].filter((row) => !sample.includes(row)));
  }
  return {
    click: ({ value }) => openPath(value[2]),
    options: () => chartOptions(axis(xLabel, (value) => compactCurrency.format(value), { scale: true }), axis("", (value) => compactCurrency.format(value)), {
      tooltip: {
        confine: true, renderMode: "richText",
        formatter: ({ value }) => `Path ${value[2]}\n${xLabel}  ${dollars.format(value[0])}\nP&L  ${dollars.format(value[1])}`,
      },
      dataZoom: zoom(),
      series: [{
        type: "scatter", symbolSize: 7, markLine: markers(true),
        itemStyle: { opacity: 0.78, color: ({ value }) => value[1] < 0 ? chartColors.loss : chartColors.accent },
        data: sample.map((row) => [row[xKey], row.pnl, row.path]),
      }],
    }),
  };
}

function rangeBins(rows, key, slots) {
  let min = Infinity, max = -Infinity;
  for (const row of rows) {
    min = Math.min(min, row[key]);
    max = Math.max(max, row[key]);
  }
  if (!rows.length) min = max = 0;
  if (min === max) {
    return { labels: [dollars.format(min)], ticks: [compactCurrency.format(min)], index: () => 0 };
  }
  const width = (max - min) / slots;
  return {
    labels: Array.from({ length: slots }, (_, i) => `${dollars.format(min + i * width)}–${dollars.format(min + (i + 1) * width)}`),
    ticks: Array.from({ length: slots }, (_, i) => compactCurrency.format(min + i * width)),
    index: (value) => Math.max(0, Math.min(slots - 1, Math.floor((value - min) / width))),
  };
}

function histogram(rows, selectBin) {
  const bins = rangeBins(rows, "pnl", 18);
  const losses = bins.labels.map(() => 0);
  const gains = bins.labels.map(() => 0);
  for (const row of rows) (row.pnl < 0 ? losses : gains)[bins.index(row.pnl)]++;
  return {
    click: ({ dataIndex }) => selectBin?.(bins.labels[dataIndex], (row) => bins.index(row.pnl) === dataIndex),
    options: () => chartOptions(axis("P&L range", undefined, { type: "category", data: bins.labels, axisLabel: { color: chartColors.label, hideOverlap: true, formatter: (_, i) => bins.ticks[i] } }), axis("", undefined, { minInterval: 1 }), {
      legend: { top: 0 },
      tooltip: { trigger: "axis", confine: true, renderMode: "richText", valueFormatter: (value) => `${value} paths` },
      dataZoom: zoom(),
      series: [
        { name: "Loss", type: "bar", stack: "paths", data: losses, color: chartColors.loss },
        { name: "Gain or break-even", type: "bar", stack: "paths", data: gains, color: chartColors.gain },
      ],
    }),
  };
}

function spotPnlHeatmap(rows, selectBin) {
  const spot = rangeBins(rows, "ending_spot", 9);
  const pnl = rangeBins(rows, "pnl", 8);
  const counts = spot.labels.map(() => pnl.labels.map(() => 0));
  for (const row of rows) counts[spot.index(row.ending_spot)][pnl.index(row.pnl)]++;
  const cells = counts.flatMap((column, x) => column.flatMap((count, y) => count ? [[x, y, count]] : []));
  return {
    click: ({ value: [x, y] }) => selectBin(`${spot.labels[x]} spot · ${pnl.labels[y]} P&L`,
      (row) => spot.index(row.ending_spot) === x && pnl.index(row.pnl) === y),
    options: () => chartOptions(
      axis("Ending spot range", undefined, { type: "category", data: spot.labels, axisLabel: { color: chartColors.label, hideOverlap: true, formatter: (_, i) => spot.ticks[i] } }),
      axis("P&L range", undefined, { type: "category", data: pnl.labels, nameGap: 61, axisLabel: { color: chartColors.label, hideOverlap: true, formatter: (_, i) => pnl.ticks[i] } }), {
        grid: { left: 82, right: 16, top: 14, bottom: 77 },
        visualMap: {
          min: 0, max: Math.max(1, ...cells.map((cell) => cell[2])), dimension: 2,
          orient: "horizontal", left: "center", bottom: 3, text: ["More paths", "Fewer paths"],
          inRange: { color: ["#ece8fc", "#7460c7", "#38248a"] },
        },
        tooltip: {
          confine: true, renderMode: "richText",
          formatter: ({ value: [x, y, count] }) => `Spot  ${spot.labels[x]}\nP&L  ${pnl.labels[y]}\n${count} ${count === 1 ? "path" : "paths"}`,
        },
        series: [{ type: "heatmap", data: cells, itemStyle: { borderColor: "#fff", borderWidth: 2 } }],
      },
    ),
  };
}
