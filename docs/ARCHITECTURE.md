# Architecture

The public `Environment.step` boundary accepts an application state, RL action, and decision time. The core delegates to frozen runtime modules for guarded progression, continuation, campaign motifs, PTP hazard, and timing. Realized events are converted to the Gold contract; decisions and explanations remain separate. Persistence adapters own local, S3, checkpoint, and optional PostgreSQL boundaries. Artifact loading is manifest-only and hash-verified before runtime initialization.

State is application-scoped. A context may later own a different application, but lifecycle prerequisites never transfer between applications. Pending future events are retained in the checkpoint and remain invisible to upstream models until realized.
