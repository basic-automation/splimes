# Security Policy

## Supported versions

splimes is pre-1.0. Only the latest published `0.x` release receives fixes.

| Version | Supported |
|---------|-----------|
| 0.1.x   | yes       |
| < 0.1   | no        |

## Reporting a vulnerability

Please **do not open a public issue** for a security problem.

Report it through GitHub's private vulnerability reporting on this repository
(Security → Report a vulnerability), which opens a channel visible only to the
maintainers.

Include what you have: the affected version or commit, a reproducer, and the impact you
think it has. You'll get an acknowledgement within a week. This is a small project, so
the fix timeline depends on severity, and we'll tell you what to expect rather than
leave you guessing.

## Scope

In scope: memory-safety problems; panics, hangs or unbounded allocation reachable from
caller-supplied points, time ranges or resolutions; wrong results returned without an
error (silent precision loss, mislabelled extrapolation); and GPU-path behaviour that
can crash the host process.

Known and tracked on the [roadmap](ROADMAP.md), so not needed as reports: GPU errors
currently panic instead of falling back to CPU, and some library paths still `unwrap`.
