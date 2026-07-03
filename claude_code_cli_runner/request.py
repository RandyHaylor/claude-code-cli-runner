"""The run_request: a generic, multimodal description of one claude -p task.

Design intent: adding a modality is DATA, not code. ``input_content`` is a LIST
of content blocks; each block is a small dataclass with a ``block_type`` and a
clear shape. Serialization to the `claude -p` stream-json user message lives in
:mod:`claude_code_cli_runner.content`, driven entirely off these shapes.

Content block shapes (all may carry the source either inline or by path):
  - TextBlock(text)
  - ImageBlock(mime_type, path=... | data_base64=...)
  - DocumentBlock(mime_type, path=... | data_base64=..., name=None)

Execution location is CONFIG, not separate code paths: ``execution_location`` is
``local_subprocess`` | ``vm_over_ssh`` | ``remote_host`` and ``ssh`` carries the
host/user/key/port details when relevant. Both transports drive the SAME
streaming runner via a build_command — proving location is config, not code.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Union

# --- multimodal input content blocks ---------------------------------------

BLOCK_TYPE_TEXT = "text"
BLOCK_TYPE_IMAGE = "image"
BLOCK_TYPE_DOCUMENT = "document"


@dataclass
class TextBlock:
    """An inline text content block."""

    text: str
    block_type: str = field(default=BLOCK_TYPE_TEXT, init=False)


@dataclass
class ImageBlock:
    """An image content block. Provide EITHER ``path`` (read+base64-encoded at
    serialization time) OR ``data_base64`` (already-encoded bytes), plus the
    ``mime_type`` (e.g. ``image/png``)."""

    mime_type: str
    path: Optional[str] = None
    data_base64: Optional[str] = None
    block_type: str = field(default=BLOCK_TYPE_IMAGE, init=False)


@dataclass
class DocumentBlock:
    """A document/file content block (e.g. a PDF). Provide EITHER ``path`` OR
    ``data_base64``, plus the ``mime_type`` (e.g. ``application/pdf``). An
    optional ``name`` labels the document for the model."""

    mime_type: str
    path: Optional[str] = None
    data_base64: Optional[str] = None
    name: Optional[str] = None
    block_type: str = field(default=BLOCK_TYPE_DOCUMENT, init=False)


ContentBlock = Union[TextBlock, ImageBlock, DocumentBlock]


# --- optional reusable (prime-once / fork-per-task) leading context ----------


@dataclass
class ReusableContext:
    """An OPTIONAL leading context chunk a caller marks as repeatable.

    ``chunk_id`` is a stable, caller-chosen unique id for this exact leading
    context. ``content`` is a LIST of the SAME content block types as
    ``RunRequest.input_content``.

    When a RunRequest carries this AND session reuse is enabled, the runner MAY
    prime a claude session with this chunk ONCE (keyed by ``chunk_id``) and then
    FORK per task so the chunk's tokens are cache-reused rather than re-sent.
    This is ALWAYS best-effort: if reuse is disabled or anything fails, the
    chunk is simply PREPENDED inline to ``input_content`` (always correct).
    """

    chunk_id: str
    content: List[ContentBlock] = field(default_factory=list)


# --- harness selection --------------------------------------------------------
# The HARNESS is the agent CLI the runner drives. Location (below) says WHERE it
# runs; harness says WHAT is launched. Both are config, not separate code paths.

HARNESS_CLAUDE_CLI = "claude_cli"
HARNESS_OPENCODE_CLI = "opencode_cli"

KNOWN_HARNESSES = (
    HARNESS_CLAUDE_CLI,
    HARNESS_OPENCODE_CLI,
)


# --- execution location config ----------------------------------------------

LOCATION_LOCAL_SUBPROCESS = "local_subprocess"
LOCATION_VM_OVER_SSH = "vm_over_ssh"
LOCATION_REMOTE_HOST = "remote_host"

KNOWN_EXECUTION_LOCATIONS = (
    LOCATION_LOCAL_SUBPROCESS,
    LOCATION_VM_OVER_SSH,
    LOCATION_REMOTE_HOST,
)


@dataclass
class SshConfig:
    """SSH transport details — pure config consumed by the vm_over_ssh /
    remote_host build_command. ``host`` may be a literal IP/hostname, or left
    None to be resolved by ``vm_name`` via the (overridable) DHCP resolver."""

    host: Optional[str] = None
    user: str = "agent-user"
    key_path: Optional[str] = None
    port: int = 22
    # Optional libvirt-DHCP lookup: when ``host`` is absent, resolve it from the
    # named VM's DHCP lease. Fully overridable; never required.
    vm_name: Optional[str] = None
    # Working directory ON the remote host to cd into before running claude.
    remote_workspace_directory: Optional[str] = None


@dataclass
class RunRequest:
    """One unified, always-streaming-capable claude -p task description.

    Fields:
      input_content: LIST of content blocks (text/image/document). Serialized
        into the stream-json user message content array.
      model: optional model id passed to claude via --model.
      workspace_directory: cwd for a local run / where the live files land
        (always on the orchestrating host, even for remote runs).
      execution_location: local_subprocess | vm_over_ssh | remote_host.
      ssh: SshConfig for the remote transports.
      dangerously_skip_permissions: explicit opt-in to
        --dangerously-skip-permissions.
      permission_mode: optional explicit permission posture (one of the claude
        --permission-mode choices). When set it takes precedence over
        dangerously_skip_permissions: the run launches WITH --permission-mode
        <mode> and WITHOUT --dangerously-skip-permissions, so the agent is not
        fully unattended (used for collaborative / manual tasks).
      session_id: optional EXPLICIT claude session id for this run, so the
        session is resumable across turns (the resume-on-reply model for
        collaborative tasks). When set, the run uses this id instead of an
        auto-generated one, and the reusable-context prime/fork path is skipped.
      resume_session: when True (with session_id set) the run CONTINUES that
        existing session via --resume <session_id> (a later turn of the same
        chat); when False with session_id set it CREATES the session via
        --session-id <session_id> (the first turn).
      extra_cli_flags: additional raw argv flags appended to the claude command.
      claude_command: the executable name/path (defaults to "claude"; tests
        point this at a stub).
      live_log_path / control_channel_path / run_status_path: optional explicit
        paths; default to the contract names under workspace_directory.
      timeout_seconds: optional overall wall-clock budget for the run.
      idle_kill_seconds: optional idle budget — the RUNNER kills the harness
        process when it streams NOTHING for this long (killing a timed-out run
        is the runner's duty; the caller only supplies the policy value).
        Waiting on an operator permission decision and operator-paused time do
        NOT count as idleness. None disables the watchdog.
      run_environment_variables: optional {NAME: value} environment variables
        set on the harness process for THIS run (merged over the serve's own
        environment). The caller uses this to hand a run its configured
        tooling facts (e.g. connection URLs/tokens) as plain environment —
        configuration, never installation.
      task_token_limit: optional hard token budget for the run — the RUNNER
        autonomously halts the harness process the moment the run's cumulative
        counted tokens (input + output + cache-creation; cache reads excluded)
        reach this number, and reports the halt as a coded result (run_state
        token_limit_halted). The caller only supplies the policy value; usage
        is always collected and reported either way. None disables the halt.
      reusable_context: optional leading ReusableContext (prime-once/fork model).
      enable_session_reuse: best-effort opt-out — when False, the reusable
        context (if any) is always PREPENDED inline; never primed/forked.
    """

    input_content: List[ContentBlock]
    workspace_directory: str
    model: Optional[str] = None
    execution_location: str = LOCATION_LOCAL_SUBPROCESS
    ssh: Optional[SshConfig] = None
    dangerously_skip_permissions: bool = False
    permission_mode: Optional[str] = None
    session_id: Optional[str] = None
    resume_session: bool = False
    extra_cli_flags: List[str] = field(default_factory=list)
    harness: str = HARNESS_CLAUDE_CLI
    claude_command: str = "claude"
    opencode_command: str = "opencode"
    live_log_path: Optional[str] = None
    control_channel_path: Optional[str] = None
    run_status_path: Optional[str] = None
    timeout_seconds: Optional[float] = None
    idle_kill_seconds: Optional[float] = None
    task_token_limit: Optional[int] = None
    run_environment_variables: dict = field(default_factory=dict)
    reusable_context: Optional[ReusableContext] = None
    enable_session_reuse: bool = True

    def __post_init__(self):
        if self.execution_location not in KNOWN_EXECUTION_LOCATIONS:
            raise ValueError(
                "unknown execution_location %r; expected one of %s"
                % (self.execution_location, ", ".join(KNOWN_EXECUTION_LOCATIONS))
            )
        if not isinstance(self.input_content, list):
            raise ValueError("input_content must be a list of content blocks")
        if self.harness not in KNOWN_HARNESSES:
            raise ValueError(
                "unknown harness %r; expected one of %s"
                % (self.harness, ", ".join(KNOWN_HARNESSES))
            )
        if self.harness == HARNESS_OPENCODE_CLI:
            # The opencode CLI has no stdio permission-escalation protocol, so an
            # operator-gated permission posture cannot be honoured — refuse loudly
            # rather than run ungated.
            if self.permission_mode:
                raise ValueError(
                    "the opencode_cli harness does not support permission_mode "
                    "(no operator permission protocol); use "
                    "dangerously_skip_permissions for a full-auto run"
                )
            # A caller-chosen CREATE id is a claude-specific contract (opencode
            # mints its own session ids); resuming a known opencode session works.
            if self.session_id and not self.resume_session:
                raise ValueError(
                    "the opencode_cli harness cannot CREATE a session with a "
                    "caller-chosen id; set resume_session=True with an existing "
                    "opencode session id, or omit session_id"
                )
            for block in self.input_content:
                if getattr(block, "block_type", None) != BLOCK_TYPE_TEXT:
                    raise ValueError(
                        "the opencode_cli harness currently supports text input "
                        "content only (got a %r block)"
                        % getattr(block, "block_type", type(block).__name__)
                    )
