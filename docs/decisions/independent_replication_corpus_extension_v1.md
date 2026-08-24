# Independent Replication Corpus Extension v1

**Status:** active outcome-blind source audit; zero model calls, zero paid
compute, and zero human-outcome reveal authorized.

The original Phase 2 viability protocol deterministically audited 40 of 57
strict-pool SocSci210 candidates. Its 17-candidate reserve remained ordered by
the same frozen SHA-256 rule. The first three reserve candidates (`h84nt`,
`r9v2d`, and `zrwjp`) all fail the current replication gates, which triggers the
old efficiency stop.

InterveneBench now has a different, explicitly prespecified purpose: build a
second prospective panel with a target of 16 and a hard minimum of 12. To avoid
silently cherry-picking attractive-looking reserve studies, this extension
opens **all 17 reserve candidates, orders 41--57**, and applies the same staged
gates to every row. No candidate may be skipped because its metadata looks
difficult, and no candidate may be retained because its title looks promising.

The exact ordered rows and selection hashes are frozen in
`data/manifests/audits/independent_replication_socsci_reserve_v1.json`. Four
rows (`mkgvp`, `waq4m`, `ux8qt`, and `gzdnf`) encountered aggregate result text in search
snippets during source lookup. Their findings were not recorded or used, but
the rows automatically fail pristine prospective eligibility and are logged in
`data/manifests/audits/independent_replication_exposure_log_v1.json`.

This extension does not weaken any scientific gate and does not authorize
inference. If the complete ordered reserve plus independently frozen external
sources cannot supply 12 outcome-clean, reconstructable, normalized decision
tasks, the replication stops before model execution rather than lowering the
standard.
