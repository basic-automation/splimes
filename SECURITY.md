# Security Policy

## Supported versions

The latest 1.x release receives fixes. 0.1 was a pre-release and is not supported;
see [MIGRATING.md](MIGRATING.md) to upgrade.

| Version | Supported |
|---------|-----------|
| 1.x     | yes       |
| 0.1.x   | no        |

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

A result outside the published numerical contract (the error bounds in the crate
documentation) is a correctness bug: report it as a normal issue, with the input that
reproduces it.
