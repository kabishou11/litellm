# Production Best Practices

The public production guide lives at [docs.litellm.ai/docs/proxy/prod](https://docs.litellm.ai/docs/proxy/prod). This page covers the dedicated-worker section of that guide against what current `main` actually implements.

## Run background jobs on a dedicated worker

**Unreleased / coming soon.** The `all`, `worker`, and `serving` values of `LITELLM_JOB_ROLE` do not control background-job registration on current LiteLLM `main`, and they are not in a released version. A replica with `LITELLM_JOB_ROLE=serving` still registers shared jobs. A replica with `LITELLM_JOB_ROLE=worker` does not become a dedicated job runner.

The proposed scheduler roles are in [PR #47878](https://github.com/BerriAI/litellm/pull/47878), which continues the closed, unmerged [PR #36618](https://github.com/BerriAI/litellm/pull/36618). Wait for a release that names scheduler-role support before deploying a serving and worker split.

On current `main` the same env var is used only by the spend collector sidecar. [`collector.py`](../../litellm/proxy/collector.py) defaults it to `collector`, and [`CollectorSettings.produces`](../../litellm/proxy/spend_tracking/spend_event_producer.py) uses that value so the collector does not forward spend events to itself. That path does not control scheduled jobs.

The environment-variable row for `LITELLM_JOB_ROLE` on the public docs site should be read the same way. `collector` is implemented. `all`, `worker`, and `serving` are not.

For multi-instance deployments today, use Redis as described on the production guide. Shared background jobs still run on every replica until scheduler roles ship.
