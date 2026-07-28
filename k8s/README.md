# Kubernetes deployment

Runs the full stack on Kubernetes: **Qdrant** (with a persistent volume), the
**Tenantq** workload (ingests the synthetic multi-tenant corpus and exposes
Prometheus metrics), and **Prometheus** scraping it.

The Tenantq container is the same `python -m tenantq.app` entrypoint used by
Docker Compose. In-cluster it talks to the Qdrant `Service` over `QDRANT_URL`
and runs with the offline hashing embedder (`TENANTQ_EMBEDDER=hash`), so pods
stay light and need no model download.

## Layout

| File | Contents |
|------|----------|
| `namespace.yaml`  | `tenantq` namespace |
| `qdrant.yaml`     | Qdrant `Deployment` + `Service` + `PersistentVolumeClaim`, with `readyz`/`livez` probes |
| `tenantq.yaml`    | Tenantq `Deployment` + `Service` (metrics on `:8001`), wired to `qdrant:6333` |
| `prometheus.yaml` | Prometheus `ConfigMap` + `Deployment` + `Service` scraping `tenantq:8001` |

## Run locally on kind

```bash
# 1. build and load the image
docker build -t tenantq:k8s .
kind create cluster --name tenantq
kind load docker-image tenantq:k8s --name tenantq

# 2. deploy
kubectl apply -f k8s/namespace.yaml
kubectl apply -f k8s/

# 3. watch it come up
kubectl -n tenantq rollout status deploy/qdrant
kubectl -n tenantq rollout status deploy/tenantq
kubectl -n tenantq get pods

# 4. see the benchmark the workload ran against in-cluster Qdrant
kubectl -n tenantq logs deploy/tenantq

# 5. inspect metrics / Prometheus
kubectl -n tenantq port-forward svc/tenantq 8001:8001    # http://localhost:8001/metrics
kubectl -n tenantq port-forward svc/prometheus 9090:9090 # http://localhost:9090
```

## Notes

- Storage for Qdrant is a `PersistentVolumeClaim`, so data survives pod
  restarts.
- To use the real dense/sparse models instead of the offline embedder, drop the
  `TENANTQ_EMBEDDER=hash` env and give the pod more memory.
- For a managed cluster (GKE/EKS/AKS), push `tenantq:k8s` to a registry and
  set `image:` in `tenantq.yaml` accordingly; everything else is portable.
