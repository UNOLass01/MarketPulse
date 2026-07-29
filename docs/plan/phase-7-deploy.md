# Phase 7 — Deployment + Documentation

**Objective:** publicly reachable and legible to a reviewer in under ten minutes.
**Complexity:** Medium · **Effort:** ~2.5 days · **Depends on:** all prior phases
**Exit criterion:** a stranger with only the public URL and the README can understand the architecture and see live predictions.

> **Do not compress this phase.** An excellent system with a poor README scores worse in a portfolio than a good system with an excellent one.

## Tasks

### Images
- [ ] Multi-stage builds; one base image with the shared package, thin per-service images on top
- [ ] Non-root user; pinned base image digests
- [ ] Built in CI, pushed to a registry

### Compose
- [ ] `docker-compose.prod.yml` — restart policies, memory limits per service, **no exposed internal ports**
- [ ] Healthcheck-gated `depends_on` with `condition: service_healthy` throughout
- [ ] Migration init container runs Alembic before app containers (never at app runtime — two replicas would race)
- [ ] 2 GB swap file configured as a safety net (with the caveat that swapping Postgres is a cliff, not a fix)

### Object storage abstraction
- [ ] All object access via a configurable S3-compatible endpoint (`endpoint_url` / `MLFLOW_S3_ENDPOINT_URL`)
- [ ] MinIO in local compose; real S3, Cloudflare R2, or Backblaze B2 in prod — **config change, not code change**
- [ ] Provider choice documented in an ADR

### Cloud
- [ ] Provision host (4 GB RAM minimum — the stack commits ~3.9 GB, which is tight and should be stated as a known risk)
- [ ] Nginx reverse proxy, TLS, basic auth on Airflow/MLflow/RabbitMQ UIs
- [ ] Firewall: inbound 22 (restricted IP) + 80/443 only. Nothing else public.
- [ ] Secrets from a parameter store via instance role — **no long-lived access keys on disk or in images**
- [ ] Log aggregation off-host (logs must survive container replacement)
- [ ] Scheduled `pg_dump` to object storage, 7-day retention
- [ ] Deploy runbook in `infrastructure/deployment.md`: pull → migrate → recreate → verify healthchecks → smoke test
- [ ] Verify rollback works: redeploy previous tag. **Test model rollback separately** — it's a registry stage transition needing no deploy at all. An untested rollback isn't a rollback.

### README (the highest-value file in the repo)
- [ ] One-paragraph what-it-does + architecture diagram
- [ ] **Why each tool was chosen**, with the alternative it beat
- [ ] Quickstart: `git clone && make up`
- [ ] **Honest observed metrics** — e.g. "52% 3-class accuracy vs 40% baseline over 12 days." A higher number would indicate leakage. Reviewers are calibrated for inflated claims; honesty is differentiating.
- [ ] Known limitations, stated proactively (single AZ, thin memory headroom, no API auth, weak signal by nature of the problem)
- [ ] **"What I'd change at real production scale"** — Kafka with symbol partitioning, TimescaleDB/ClickHouse, feature store, Prometheus/Grafana, ECS/EKS, Terraform, shadow deployment, multi-AZ — each with its *trigger condition*
- [ ] If you deployed somewhere other than AWS, say so. Claiming AWS and being asked to walk through the console unravels badly.

### Remaining docs
- [ ] ADRs: `0002-ec2-over-lambda`, `0003-postgres-over-timeseries-db`, `0004-scheduled-over-drift-triggered-retraining`, `0005-three-class-target-with-deadband`, `0006-features-computed-once-at-ingestion`, `0007-object-storage-provider`
- [ ] `notebooks/README.md` stating explicitly that notebooks are exploratory and non-authoritative
- [ ] 2-minute demo video: data flowing → Airflow run → MLflow promotion → API response changing version → drift dashboard

## Tests
- [ ] `make test-e2e` green against the prod compose profile
- [ ] Smoke test script: health, ready, one prediction, dashboard loads
- [ ] Fresh-clone test: clone into an empty dir, `make up`, system works from scratch
- [ ] Restart the whole stack → all state survives (volumes, model, data)
- [ ] Restore from a `pg_dump` backup into a clean DB
- [ ] Internal ports confirmed unreachable from outside

## Cost discipline
- [ ] Run 4–5 days to accumulate real drift and accuracy data, capture screenshots + video, then **tear down**
- [ ] "I tore it down rather than pay to keep a demo warm — here's the video and one-command setup" is itself a cost-awareness signal
