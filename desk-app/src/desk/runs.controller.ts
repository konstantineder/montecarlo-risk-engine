import { Controller, Get, NotFoundException, Param } from "@nestjs/common";
import { RunsService } from "./runs.service";

@Controller("api/runs")
export class RunsController {
  constructor(private readonly runs: RunsService) {}

  @Get()
  list() {
    return this.runs.list();
  }

  @Get(":id")
  one(@Param("id") id: string) {
    const run = this.runs.get(id);
    if (!run) throw new NotFoundException(`No run ${id}`);
    return run;
  }

  @Get(":id/paths")
  paths(@Param("id") id: string) {
    const rows = this.runs.paths(id);
    if (!rows) throw new NotFoundException(`No run ${id}`);
    return rows;
  }

  @Get(":id/paths/:index")
  path(@Param("id") id: string, @Param("index") index: string) {
    const row = this.runs.path(id, Number(index));
    if (!row) throw new NotFoundException(`No path ${index} in ${id}`);
    return row;
  }
}
