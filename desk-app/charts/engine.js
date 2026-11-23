import { init, use } from "echarts/core";
import { BarChart, HeatmapChart, LineChart, ScatterChart } from "echarts/charts";
import {
  DataZoomComponent, GridComponent, LegendComponent, MarkLineComponent,
  TooltipComponent, VisualMapComponent,
} from "echarts/components";
import { CanvasRenderer } from "echarts/renderers";

use([
  BarChart, HeatmapChart, LineChart, ScatterChart, CanvasRenderer,
  DataZoomComponent, GridComponent, LegendComponent, MarkLineComponent,
  TooltipComponent, VisualMapComponent,
]);

export { init };
