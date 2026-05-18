# Increasing-failure-of-post-heatwave-rainfall-recovery-in-the-northwestern-United-States
Data and code for analysing post-heatwave rainfall recovery failure after dry heatwave events across the contiguous United States.
# Event-scale Heatwave Recovery and Mechanism Analysis

This repository contains the code used to identify three-dimensional heatwave events, extract post-heatwave rainfall windows, and analyse event-scale rainfall recovery, structural changes, regional heterogeneity, and land–atmosphere mechanism diagnostics over the contiguous United States.

The workflow is designed for reproducible manuscript figure generation for an event-based drought–heatwave recovery study. It combines 3D connected-component heatwave tracking, post-event rainfall recovery diagnostics, structural standardization, regional and spatial trend analysis, mechanism-proxy construction, Northwest-focused mechanism figures, and precipitation-product sensitivity tests.

---

## 1. Project Overview

The analysis asks how rainfall recovery after heatwave events varies across space, time, event structure, and land–atmosphere state.

The repository supports four major tasks:

1. **Heatwave event extraction**
   - Identifies heatwave objects using 3D connected-component labeling.
   - Uses 26-neighbour spatiotemporal connectivity.
   - Detects heatwave events from the heatwave core only, without allowing post-event rainfall windows to affect event definition.
   - Exports per-event core, full event-window, event-level post-window, grid-level post-window, and annual summary files.

2. **Result 1: Post-heatwave rainfall recovery**
   - Calculates day-10 rainfall recovery probability.
   - Estimates first-recovery lag and recovery hazard.
   - Maps spatial recovery probability.
   - Quantifies regional heterogeneity across seven U.S. climate regions.
   - Tests robustness across rainfall thresholds, footprint thresholds, and recovery windows.

3. **Result 2: Structural standardization**
   - Separates observed changes in no-recovery probability from changes associated with event duration, heat excess, footprint size, seasonality, and regional composition.
   - Calculates binary no-recovery outcomes and continuous rain-return metrics.
   - Produces annual observed and standardized series, regional decomposition, spatial trend maps, rolling-window sensitivity, and cluster-bootstrap summaries.

4. **Result 3: Mechanism bridge and diagnostic closure**
   - Links recovery failure to continuous rain-return loss, event structure, land end-state, and atmospheric support.
   - Builds event-end mechanism proxies, including:
     - Ridging–Subsidence Index (RSI)
     - Moisture-Support Deficit Index (MSDI)
     - Land-memory proxy based on soil moisture and Bowen ratio
   - Produces Northwest-focused blocking, moisture, land-surface, and mechanism-closure figures.

---

## 2. Main Scientific Definitions

### Heatwave event

Heatwave events are defined from gridded daily heatwave-core records. The extraction script uses:

- `heat3 == 1` as the heatwave-core indicator.
- Minimum event duration: 3 days.
- 3D connected-component labeling with 26-neighbour connectivity in longitude, latitude, and time.
- Event objects are identified using heatwave-core records only.

### Post-event rainfall recovery

Most downstream analyses define rainfall recovery within the post-event day 1–10 window as:

- precipitation ≥ 1 mm day⁻¹, and
- rainy cells covering at least 25% of the original heatwave footprint.

Some extraction outputs also retain lag 0, depending on the `INCLUDE_LAG0` setting in the event-extraction script. Downstream Result 1 and Result 2 analyses generally use post-event days 1–10.

### Region assignment

Events are assigned to one of seven U.S. climate regions:

- Northwest
- Northern Great Plains
- Midwest
- Northeast
- Southwest
- Southern Great Plains
- Southeast

Region assignment is based on the majority of heatwave-core footprint cells mapped to U.S. state polygons. Centroid-based assignment is used only as a fallback or tie-breaker.
