# Vendored third-party code

Served from this server so the dashboard loads no script from another host
(see the Content-Security-Policy in main.py). A CDN is a party that can change
the code your SOC console runs; this file can only change in a commit.

| File | Upstream | Licence |
|---|---|---|
| `chart.umd.min.js` | chart.js **4.4.1**, `package/dist/chart.umd.js` from the npm tarball | MIT (`chart.js.LICENSE.md`) |

Provenance, checked 2026-10-31:

* tarball integrity `sha512-C74QN1bxwV1v2PEujhmKjOZ7iUM4w6BWs23Md/6aOZZSlwMzeCIDGuZay++rBgChYru7/+QFeoQW0fQoP534Dg==` matched the registry's `dist.integrity`
* the npm registry's ECDSA signature over `chart.js@4.4.1:<integrity>` verified
  against https://registry.npmjs.org/-/npm/v1/keys
* the trailing `//# sourceMappingURL=chart.umd.js.map` line was removed (the
  map is not shipped); nothing else was changed

sha256 of the file as committed: `0ee28337f25838a7a5d6b1e8b2b02279ab10e80e17c217a99893a7a717f2ba05`

The cdnjs copy the dashboard used before is a separate re-minification by cdnjs
(different identifiers, no licence header), not the upstream build.

To upgrade: download the new tarball from the registry, repeat both checks,
replace the file, update this table and the version assertion in
tests/test_browser_hardening.py.
