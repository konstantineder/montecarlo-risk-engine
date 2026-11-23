import { Body, Controller, Get, NotFoundException, Param, Post, Query, Res } from "@nestjs/common";
import { RunsService } from "./runs.service";

@Controller("api")
export class RunsController {
  constructor(private readonly runs: RunsService) {}

  @Get("runs")
  list(@Query("limit") limit?: string, @Query("offset") offset?: string) {
    return this.runs.list(Number(limit) || 20, Number(offset) || 0);
  }

  @Get("runs/:id")
  one(@Param("id") id: string) {
    const run = this.runs.get(id);
    if (!run) throw new NotFoundException(`No run ${id}`);
    return run;
  }

  @Get("runs/:id/paths")
  paths(@Param("id") id: string) {
    const rows = this.runs.paths(id);
    if (!rows) throw new NotFoundException(`No run ${id}`);
    return rows;
  }

  @Get("runs/:id/series")
  series(@Param("id") id: string) {
    const rows = this.runs.series(id);
    if (!rows) throw new NotFoundException(`No run ${id}`);
    return rows;
  }

  @Get("runs/:id/export.json")
  exportJson(@Param("id") id: string, @Res() res: any) {
    if (!this.runs.get(id)) throw new NotFoundException(`No run ${id}`);
    this.runs.streamJson(id, res, "full");
  }

  @Get("runs/:id/export.mtm.json")
  exportMtm(@Param("id") id: string, @Res() res: any) {
    if (!this.runs.get(id)) throw new NotFoundException(`No run ${id}`);
    this.runs.streamJson(id, res, "mtm");
  }

  @Get("runs/:id/export.book.json")
  exportBook(@Param("id") id: string, @Res() res: any) {
    if (!this.runs.get(id)) throw new NotFoundException(`No run ${id}`);
    this.runs.streamJson(id, res, "book");
  }

  @Get("runs/:id/export.csv")
  exportCsv(@Param("id") id: string, @Res() res: any) {
    if (!this.runs.get(id)) throw new NotFoundException(`No run ${id}`);
    this.runs.streamCsv(id, res);
  }

  @Get("runs/:id/paths/:index")
  async path(@Param("id") id: string, @Param("index") index: string) {
    const row = await this.runs.path(id, Number(index));
    if (!row) throw new NotFoundException(`No path ${index} in ${id}`);
    return row;
  }

  @Get("rules")
  rules() {
    return this.runs.catalogue();
  }

  @Post("simulate")
  simulate(@Body() body: Record<string, unknown>) {
    return this.runs.simulate(body || {});
  }

  @Get("simulation/config")
  configuration() { return this.runs.configuration(); }

  @Post("simulation/preview")
  preview(@Body() body: Record<string, unknown>) { return this.runs.preview(body || {}); }

  @Get("jobs")
  jobs() { return this.runs.jobs(); }

  @Post("jobs/batches/:id/combine")
  combineBatch(@Param("id") id: string) { return this.runs.combineBatch(id); }

  @Post("runs/:id/reread")
  reread(@Param("id") id: string, @Body() body: Record<string, string>) {
    const job = this.runs.reread(id, body || {});
    if (!job) throw new NotFoundException(`No run ${id}`);
    return job;
  }

  @Post("jobs/:id/pause")
  pause(@Param("id") id: string) { return this.runs.pause(id); }

  @Post("jobs/:id/resume")
  resume(@Param("id") id: string) { return this.runs.resume(id); }

  @Get("optimizer/context/:run/:path")
  optimizerContext(@Param("run") run: string, @Param("path") path: string) { return this.runs.optimizer.context(run, Number(path)); }

  @Post("optimizer/:run/:path/preview")
  optimizerPreview(@Param("run") run: string, @Param("path") path: string, @Body() body: Record<string, unknown>) { return this.runs.optimizer.preview(run, Number(path), body || {}); }

  @Post("optimizer/:run/:path/start")
  optimize(@Param("run") run: string, @Param("path") path: string, @Body() body: Record<string, unknown>) { return this.runs.optimizer.start(run, Number(path), body || {}); }

  @Get("optimizations/:id")
  optimization(@Param("id") id: string) { return this.runs.optimizer.result(id); }

  @Post("optimizations/:id/replay")
  replay(@Param("id") id: string, @Body() body: Record<string, unknown>) { return this.runs.optimizer.replay(id, body || {}); }

  @Post("optimizations/:id/save")
  saveOptimized(@Param("id") id: string, @Body() body: Record<string, unknown>) { return this.runs.optimizer.save(id, body || {}); }

  @Get("configurations")
  configurations() { return this.runs.optimizer.configurations(); }

  @Get("jobs/:id")
  job(@Param("id") id: string) {
    const job = this.runs.job(id);
    if (!job) throw new NotFoundException(`No job ${id}`);
    return job;
  }
}
