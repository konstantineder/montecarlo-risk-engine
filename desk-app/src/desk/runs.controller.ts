import { Body, Controller, Get, NotFoundException, Param, Post } from "@nestjs/common";
import { RunsService } from "./runs.service";

@Controller("api")
export class RunsController {
  constructor(private readonly runs: RunsService) {}

  @Get("runs")
  list() {
    return this.runs.list();
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

  @Get("runs/:id/paths/:index")
  path(@Param("id") id: string, @Param("index") index: string) {
    const row = this.runs.path(id, Number(index));
    if (!row) throw new NotFoundException(`No path ${index} in ${id}`);
    return row;
  }

  @Post("simulate")
  simulate(@Body() body: Record<string, string>) {
    return this.runs.simulate(body || {});
  }

  @Get("jobs/:id")
  job(@Param("id") id: string) {
    const job = this.runs.job(id);
    if (!job) throw new NotFoundException(`No job ${id}`);
    return job;
  }
}
