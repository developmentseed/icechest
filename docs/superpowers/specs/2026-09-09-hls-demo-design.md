# HLS integration demo

**Date:** 2026-09-09
**Status:** approved, not yet implemented

## Purpose

Demonstrate icechest end to end against real data: take STAC metadata records
for HLS granules, build VirtualiZarr representations of the COGs those records
describe, and publish both halves in a single Icechunk commit — 50 to 100
granules at a time.

The deliverable is a module whose stages can be run individually from notebook
cells, so each step of the pipeline can be shown on its own. It runs against a
local Icechunk store now and is expected to move to cloud infrastructure later,
unchanged in shape.

## Goals

- Read granule metadata from the MAAP HLS STAC-geoparquet archive.
- Write it into an icechest table whose schema is the archive's schema.
- Virtualize every asset of every selected granule, at every resolution level
  the COG carries.
- Declare the `multiscales`, `proj:` and `spatial:` Zarr conventions on the
  written groups, populated from the STAC record.
- Commit metadata and virtual references together, per batch.
- Keep every stage independently callable, with the network-touching stages
  runnable one granule at a time.

## Non-goals

- Cloud deployment. The module must not *prevent* it — hence no local-only
  assumptions in the reference URLs — but no deployment code is in scope.
- Concatenating granules into cubes. Each granule stands alone; see "One group
  per granule" below.
- Re-ingest and update flows. `tx.overwrite` makes them straightforward later,
  but the demo only appends.
- HLSS30. `open_archive` takes a collection parameter, but the default band list
  is HLSL30's and S30's differing band names are not handled.

## Background

Two data sources, both verified against live data while writing this spec.

### The metadata archive

`s3://nasa-maap-data-store/file-staging/nasa-map/hls-stac-geoparquet-archive/v2/HLSL30_2.0/iceberg/metadata/latest.metadata.json`
is a static Iceberg `metadata.json` over hive-partitioned GeoParquet. It holds
15,944,395 records in 162 data files, with an empty partition spec.

PyIceberg reads it with `StaticTable.from_metadata(location)` — no catalog
service, no DuckDB. That matters beyond convenience: the archive is an Iceberg
table resolved without a catalog, which is the same shape icechest itself
argues for.

Its schema is STAC fields flattened at the top level: `id`, `collection`,
`datetime`, `eo:cloud_cover`, `proj:epsg`, `proj:shape`, `proj:transform`,
`bbox` (a struct), `geometry` (binary WKB), and an `assets` struct with fixed
band keys (`B01`–`B07`, `B09`–`B11`, `Fmask`, `SZA`, `SAA`, `VZA`, `VAA`,
`thumbnail`), each carrying `href`, `type`, `roles` and `eo:bands`.

**The proj fields are only populated on recent records.** A 2000-row sample of
the oldest data (2013) had `proj:shape` and `proj:transform` null in every row;
a 3000-row sample of `datetime >= 2026-01-01` had them populated in every row.
This is why the selection filter has a date floor rather than a date default.

### The assets

Asset hrefs point at LP DAAC over HTTPS:

```
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/HLSL30.020/
  HLS.L30.T33MVR.2013289T091221.v2.0/HLS.L30.T33MVR.2013289T091221.v2.0.B04.tif
```

Those objects are also reachable as `s3://lp-prod-protected/HLSL30.020/...`
with temporary credentials from LP DAAC's `s3credentials` endpoint, which
authenticates with Earthdata Login.

Reading the IFD chain of one B04 COG confirms the pyramid is real:

```
IFD 0: 3660x3660 tile=256x256
IFD 1: 1830x1830 tile=128x128 subfile=1
IFD 2:  915x915  tile=128x128 subfile=1
IFD 3:  458x458  tile=128x128 subfile=1
IFD 4:  229x229  tile=128x128 subfile=1
```

Five levels, factor 2 each step, with georeferencing (`ModelPixelScale`,
`ModelTiepoint`, `GeoAsciiParams`) on IFD 0 only. The level count is read per
asset rather than assumed, because Fmask and the angle bands need not match
B04's depth.

## Architecture

### Placement

`src/icechest/demo/`, a subpackage of icechest, with its extra dependencies
behind an optional `demo` extra. `icechest`'s five core dependencies do not
change, `import icechest` keeps working without the extra, and demo code imports
as `from icechest.demo import select_granules`.

### Modules

| Module | Entry points | Network |
| --- | --- | --- |
| `source.py` | `open_archive(collection) -> StaticTable`; `select_granules(table, *, bbox, tile, datetime, limit) -> pa.Table` | MAAP bucket |
| `assets.py` | `asset_urls(row, bands) -> dict[str, str]` | none |
| `credentials.py` | `lpdaac_credentials()`; `virtual_chunk_container()`; `object_store_registry()` | Earthdata |
| `conventions.py` | `granule_attrs(*, epsg, shape, transform, levels) -> dict` | none |
| `virtualize.py` | `virtual_granule(row, *, registry, bands) -> GranuleArrays` | LP DAAC |
| `store.py` | `open_store(path) -> HybridRepo`; `ensure_table(repo, schema)` | local disk |
| `ingest.py` | `ingest_batch(repo, rows, *, registry, bands) -> BatchResult` | all of the above |

The boundary that matters most: `assets.py` and `conventions.py` touch no
network and hold most of the fiddly correctness, so they are unit-testable
offline. Every other stage can be driven with a single granule before a batch is
attempted.

### Data flow

```
select_granules      -> pa.Table of rows in the archive's schema
  for each row:
    asset_urls       -> {band: s3://lp-prod-protected/...}
    virtual_granule  -> {band: {level: virtual dataset}} + level counts
    granule_attrs    -> convention attributes for the band group
    write            -> ds.vz.to_icechunk(store=tx.session.store,
                                          group=f"/{id}/{band}/multiscales",
                                          mode="a")
  tx.append("granules", successful rows + array_path)
  tx.commit()        -> one snapshot carrying both halves
```

## Decisions

### Source schema verbatim, plus one column

The table is created from the archive's own PyIceberg `Schema` with a single
added `array_path` string column. No subset, no remodelling: curating a schema
would reintroduce exactly the bespoke metadata modelling icechest exists to
avoid, and the archive's schema is already an Iceberg schema we can pass
straight to `tx.create_table`.

`array_path` holds the granule's group path (`/{id}`). The alternative — deriving
it from `id` by convention — would keep the table byte-identical to the source,
but leaves the link implicit; an explicit column is self-describing and costs one
field.

### Selection has a floor, not a default

`select_granules` always applies `datetime >= 2026-01-01` **and**
`NotNull("proj:transform")`. Optional `bbox`, `tile`, `datetime` and `limit`
narrow further; a caller-supplied `datetime` range is intersected with the
floor, never substituted for it.

Both conditions are required. The date alone is not a guarantee, and a null
transform would silently produce a `spatial:` convention with missing
dimensions and transform — a half-declared convention is worse than none.

### One group per granule

Each granule owns `/{id}`, holding one group per band, holding one array per
resolution level: `/{id}/{band}/multiscales/{level}`. A row maps to exactly one
group.

Concatenating granules into per-tile cubes was considered and rejected for this
demo: it requires same-grid granules, and it turns the row-to-array mapping from
a path into a path plus an index. Heterogeneous granules coexisting under a
queryable index is the premise icechest exists to serve.

### All assets, all levels

Default band list is all fifteen non-thumbnail assets: `B01`–`B07`, `B09`–`B11`,
`Fmask`, `SZA`, `SAA`, `VZA`, `VAA`. The list is a parameter.

Every IFD is virtualized, with the level count read from each asset's own chain.
At roughly five levels per asset that is about 75 virtual arrays per granule, so
a 75-granule batch stages on the order of 5,600 virtual arrays from about 1,100
asset header reads. This is not instant; notebook stepping should use a small
`limit`.

### Assets are read as `s3://`, and that is what gets stored

`asset_urls` rewrites the LP DAAC HTTPS href to
`s3://lp-prod-protected/HLSL30.020/...`. Credentials come from LP DAAC's
`s3credentials` endpoint, authenticated from `~/.netrc`.

This choice is durable, not just operational: the URL recorded in a virtual
reference is the URL every future reader must resolve. `s3://` is the form that
works in-region at scale, and it is what the Icechunk virtual chunk container is
declared against.

Two mechanics follow:

- The repository declares a `VirtualChunkContainer` for the
  `s3://lp-prod-protected/` prefix via `RepositoryConfig.set_virtual_chunk_container`,
  and readers authorize it through `containers_credentials`.
- Credentials expire hourly, so they are supplied through
  `icechunk.s3_refreshable_credentials(get_credentials=...)`. That callable
  **must be picklable**, so `lpdaac_credentials` is a module-level function, not
  a closure over configuration.

### Conventions come from the record

These attributes are the only georeferencing the written groups carry, so they
have to satisfy the conventions' own published schemas — a reader that cannot
validate them cannot use them. The authority is the schema at each convention's
`v0.1` tag under `github.com/zarr-conventions/`, fetched and checked while
writing this section. Where this document previously disagreed with those
schemas, the schemas win; the corrections are called out below.

`granule_attrs` is pure. Given `proj:epsg`, `proj:shape`, `proj:transform` and a
level count, it returns:

- `zarr_conventions`: the three convention metadata objects. Each schema fixes
  every field of its own object as a `const` and sets
  `additionalProperties: false`, so these must be reproduced exactly and carry
  nothing extra:

  | name | uuid | urls |
  | --- | --- | --- |
  | `multiscales` | `d35379db-88df-4056-af3a-620245f8e347` | `zarr-conventions/multiscales` @ `v0.1` |
  | `proj` | `f17cb550-5864-4468-aeb7-f3180cfb622f` | `zarr-conventions/proj` @ `v0.1` |
  | `spatial` | `689b58e2-cf7b-45e0-9fff-9cfc0883d6b4` | `zarr-conventions/spatial` @ `v0.1` |

  **Corrected:** the names carry no trailing colon — `proj`, not `proj:`. The
  published tag is `v0.1`, not `v1`, so the previously recorded `refs/tags/v1/`
  URLs 404. And the proj convention lives at `zarr-conventions/proj`; the
  `zarr-experimental/geo-proj` location it was recorded under now only
  redirects.

- `proj:code` from `proj:epsg`, as the authority-qualified string —
  `"EPSG:32620"`, matching the schema's `EPSG:32633` example.
- `spatial:shape` from `proj:shape`, as `[height, width]`.
  **Corrected:** this was previously written to `spatial:dimensions`, which the
  schema defines as the *names* of the two spatial dimensions, typed as strings
  (`["y", "x"]`). Putting pixel counts there left a reader looking for the grid
  size finding nothing, and a reader reading dimension names finding integers.
- `spatial:dimensions` as `["y", "x"]` — the row-major names the schema asks
  for.
- `spatial:transform` from the first six elements of `proj:transform`. The
  archive stores a nine-element row-major affine whose last row is `0, 0, 1`;
  the convention wants the six that carry the affine, `[a, b, c, d, e, f]`,
  mapping array indices to coordinates. This one was already right.
- `spatial:bbox` computed from the transform and shape:
  `xmin = transform[2]`, `ymax = transform[5]`,
  `xmax = xmin + cols * transform[0]`, `ymin = ymax + rows * transform[4]`
  (`transform[4]` is negative). This is exact and needs no reprojection library,
  which is why the geographic `bbox` struct is not used for it.
- `multiscales.layout`: one entry per level. Level 0 is `{"asset": "0"}`; each
  subsequent level carries `derived_from` naming the level above it and
  `transform: {"scale": [2, 2]}`.
  **Corrected:** a layout item's `transform` is an object with `scale` and
  `translation`, and it is defined *relative to `derived_from`* — so the value
  is the factor-of-two step between adjacent levels, not the level's absolute
  affine. The previously specified flat six-element absolute transform is not a
  valid layout transform at all, and `factors` is not a defined key; it survived
  only because the schema allows additional properties. A per-level absolute
  affine does have a defined home if one is wanted later: the spatial convention
  permits `spatial:shape` and `spatial:transform` as overrides inside a layout
  item.

`asset` and `derived_from` are paths relative to the group holding the
`multiscales` attributes, so where those attributes are written determines
whether the declared paths resolve. Whichever group they land on, the paths
must reach the arrays actually written.

The attributes land on the band group, `/{id}/{band}`, so the layout's paths
are `multiscales/0`, `multiscales/1` and so on. That keeps the band — the node
a reader lands on for one asset — self-describing without descending into the
pyramid, and keeps the attributes off any group `to_icechunk` itself opens.
`VirtualTIFF(ifd=n)` names its single variable `str(n)` and `to_icechunk`
writes variables *inside* the group it is given, so all levels go into one
shared `multiscales` group and the variable name supplies the level; naming
the level in the group path too would bury each array one node below
everything that references it.

### Metadata and data must agree

`virtual_granule` returns a `GranuleArrays` — the per-band, per-level virtual
datasets, the level count found for each band, and the shape read from IFD 0 —
and it compares that shape against the row's `proj:shape`, failing the granule
when they disagree. The whole premise of this store is that
the metadata and the arrays cannot drift apart; a granule whose record already
contradicts its data is exactly what should not be published.

### One commit per batch, failures skipped

`ingest_batch` opens one transaction, writes each granule's virtual references
into that transaction's session, appends the successful rows once, and commits
once. Virtual writes go to `tx.session.store`, so they are staged in the same
session as the table pointer and published by the same commit —
`ds.vz.to_icechunk` stages into the session without committing it, which is what
makes a single commit per batch possible. The batch test asserts this directly
by counting snapshots.

A granule that fails — missing asset, unparseable header, shape mismatch —
contributes neither arrays nor a row, and is recorded in
`BatchResult.skipped` with its reason. The metadata/array invariant still holds
exactly: it is about a granule's row and its arrays landing together, not about
all hundred granules landing. Aborting a whole batch for one unavailable COG
would repeat all the work on retry, which is the wrong behaviour at archive
scale.

`BatchResult` carries the snapshot id, the committed granule ids, and the
skipped ids with reasons.

### Concurrency is inherited, not built

Two batches writing disjoint `/{id}` groups rebase through
`BasicConflictSolver`, and their table appends replay through the existing
intent machinery. Parallel workers need no new code in this module.

## Failure handling

| Failure | Handling |
| --- | --- |
| Asset href missing from the record | Skip that band; granule proceeds with the rest |
| COG header unreadable | Skip the granule, record the reason |
| IFD 0 shape disagrees with `proj:shape` | Skip the granule, record both shapes |
| Expired LP DAAC credentials mid-batch | Surfaces as read failures; refreshable credentials cover reads, but a batch running past the credential lifetime should be split. Documented, not worked around |
| Icechunk commit conflict | Handled by icechest's existing retry and replay |
| Every granule in a batch fails | `ingest_batch` raises its own error naming the batch and the per-granule reasons, rather than letting the transaction's generic "made no changes" surface |

## Testing

No network in the default suite.

- `assets.py`: href rewriting, including a href that does not match the expected
  LP DAAC prefix.
- `conventions.py`: attribute construction against a known transform and shape,
  including the projected-bbox arithmetic and the multiscales layout for a
  five-level pyramid. This is where most correctness lives and it is entirely
  pure.
- `source.py`: that the selection filter includes both the date floor and the
  not-null condition. Filter construction is checked without a scan.
- `ingest.py`: batch assembly against a stub virtualizer that can be told to
  fail on a given granule — this is how skip-and-report is covered without
  needing a genuinely broken COG. Asserts the committed snapshot carries exactly
  the successful rows and their groups.
- An opt-in `@pytest.mark.network` integration test ingesting two or three real
  granules end to end, reading both halves back from one snapshot.

The notebook is a deliverable, not a test target.

## Deliverables

- `src/icechest/demo/` with the seven modules above.
- `examples/hls_ingest.ipynb`, alongside the existing `examples/concurrent_writers.py`,
  walking through the stages one cell at a time.
- An optional `demo` extra in `pyproject.toml` carrying `virtualizarr>=2.7`,
  `virtual-tiff>=0.5`, and `obstore>=0.11`.

## Risks

**Runtime.** About 1,100 asset header reads per 75-granule batch. If that proves
too slow to be a usable demo, the mitigation is a smaller default `limit`, not a
design change.

**Credential lifetime.** LP DAAC credentials last an hour. A batch that outruns
them fails partway; the granules that already succeeded are skipped-with-reason
rather than lost, but the batch is smaller than asked for.
