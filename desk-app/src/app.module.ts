import { Module } from "@nestjs/common";
import { RunsModule } from "./desk/runs.module";
import { HealthController } from "./health.controller";

@Module({
  imports: [RunsModule],
  controllers: [HealthController],
})
export class AppModule {}
