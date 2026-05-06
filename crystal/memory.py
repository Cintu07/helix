"""
HelixMemory — unified orchestrator for the full crystal suite.

Single entry point that chains together:
  MultiModalFusion   (text + image + audio absorption)
  TemporalPhaseIndex (random-access memory timeline)
  AffectiveEncoder   (emotional state tracking)
  PhaseCollapseRegister (permanent binary flags)
  ContextDistiller   (compression stats)
  PhaseDiff / PhaseVersionTracker (memory versioning)
  PhiCrypt           (encrypted .hxe storage)

Usage:
    mem = HelixMemory(hidden_size=64, passphrase="secret")
    mem.absorb(text=embedding, valence=0.8, arousal=0.3)
    mem.save("session")          # writes session.hxe (encrypted)

    mem2 = HelixMemory(hidden_size=64, passphrase="secret")
    mem2.load("session.hxe")
    features = mem2.recall()     # (hidden_size * harmonics * 2,) tensor
"""

import os
import torch

from .multimodal import MultiModalFusion
from .temporal_index import TemporalPhaseIndex
from .affective import AffectiveEncoder
from .phase_collapse import PhaseCollapseRegister
from .distillation import ContextDistiller
from .phase_diff import PhaseDiff, PhaseVersionTracker
from .phicrypt import PhiCrypt


class HelixMemory:

    def __init__(
        self,
        hidden_size=64,
        unified_dim=128,
        harmonics=None,
        snapshot_interval=10,
        passphrase=None,
        num_flags=32,
        affective_neurons=8,
    ):
        harmonics = harmonics or [1, 2, 4, 8]
        self.hidden_size = hidden_size
        self.harmonics = harmonics
        self.passphrase = passphrase
        self._step = 0
        self._last_phi = torch.zeros(hidden_size)

        self.fusion = MultiModalFusion(
            hidden_size=hidden_size,
            unified_dim=unified_dim,
            harmonics=harmonics,
        )
        # full_state=True → phase lives on real line, not wrapped → circular=False
        self.tpi = TemporalPhaseIndex(
            hidden_size=hidden_size,
            snapshot_interval=snapshot_interval,
            circular=False,
        )
        self.affect = AffectiveEncoder(
            hidden_size=hidden_size,
            affective_neurons=affective_neurons,
        )
        self.flags = PhaseCollapseRegister(num_flags=num_flags)
        self.distiller = ContextDistiller(
            input_size=unified_dim,
            hidden_size=hidden_size,
            harmonics=harmonics,
        )
        self.differ = PhaseDiff()
        self.tracker = PhaseVersionTracker(self.differ)
        self.crypt = PhiCrypt() if passphrase else None

    # ------------------------------------------------------------------
    # Core API
    # ------------------------------------------------------------------

    def absorb(self, text=None, image=None, audio=None, valence=None, arousal=None):
        """Feed one turn of data. Any combination of modalities is valid."""
        if text is not None:
            self.fusion.absorb_text(text)
        if image is not None:
            self.fusion.absorb_image(image)
        if audio is not None:
            self.fusion.absorb_audio(audio)
        if valence is not None:
            self.affect.encode_sentiment(valence, arousal if arousal is not None else 0.5)

        phi = self.fusion.recall_compact()
        self.tpi.record(self._step, phi)
        self._last_phi = phi
        self._step += 1

    def recall(self):
        """Full harmonic feature vector from current phase state."""
        return self.fusion.recall()

    def recall_compact(self):
        """Raw phase angles — most compact form."""
        return self._last_phi.clone()

    def recall_at(self, step):
        """Phase state at a specific past step (temporal random access)."""
        return self.tpi.recall_at(step)

    def search(self, query_phi, top_k=5):
        """Find past steps whose phase state is most similar to query_phi."""
        return self.tpi.search(query_phi, top_k)

    # ------------------------------------------------------------------
    # Affect
    # ------------------------------------------------------------------

    def affect_state(self):
        """Current emotional state: {'valence': float, 'arousal': float, 'label': str}"""
        return self.affect.decode_sentiment()

    def affect_trajectory(self):
        """Full emotional history across all absorbed turns."""
        return self.affect.emotional_trajectory()

    # ------------------------------------------------------------------
    # Phase velocity
    # ------------------------------------------------------------------

    def phase_velocity(self):
        """
        How fast the memory is changing right now.
        High = lots of new information being absorbed.
        Low  = redundant or repeated information.
        """
        if self._step < 2:
            return None
        return self.tpi.phase_velocity_at(self._step - 1)

    # ------------------------------------------------------------------
    # Versioning
    # ------------------------------------------------------------------

    def commit(self, message=""):
        """Snapshot the current phase state as a named version."""
        return self.tracker.commit(self._last_phi, message)

    def rollback(self, version):
        """Restore the memory to a previously committed version."""
        phi = self.tracker.rollback(version)
        self._last_phi = phi
        return phi

    def diff(self, version_a=None, version_b=None):
        """
        Diff two committed versions. If versions not specified, diffs the last two.
        Returns a PhaseChangeSet with .summary(), .num_major_changes(), etc.
        """
        if version_a is not None and version_b is not None:
            return self.tracker.diff_versions(version_a, version_b)
        if self.tracker.num_versions() < 2:
            raise ValueError("Need at least 2 committed versions to diff.")
        n = self.tracker.current_version
        return self.tracker.diff_versions(n - 1, n)

    def version_log(self):
        return self.tracker.log()

    # ------------------------------------------------------------------
    # Permanent flags
    # ------------------------------------------------------------------

    def register_flag(self, name, idx):
        """Register a named permanent binary flag at a given index."""
        self.flags.register_flag(idx, name)

    def set_flag(self, name):
        """Permanently set a named flag. Irreversible."""
        self.flags.collapse_named(name)

    def get_flag(self, name):
        """Check if a named flag is set."""
        return self.flags.query_named(name)

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path):
        """
        Save crystal to disk. Encrypts to .hxe if passphrase was set,
        otherwise writes plain .hx.
        Returns the final file path written.
        """
        hx_path = path if path.endswith(".hx") else path + ".hx"
        self.fusion.export(hx_path)

        if self.crypt and self.passphrase:
            enc_path = hx_path[:-3] + ".hxe"
            self.crypt.encrypt_file(hx_path, enc_path, self.passphrase)
            os.remove(hx_path)
            return enc_path
        return hx_path

    def load(self, path):
        """
        Load crystal from disk. Decrypts .hxe if passphrase was set.
        Accepts both .hx and .hxe paths.
        """
        if path.endswith(".hxe"):
            if not (self.crypt and self.passphrase):
                raise ValueError("passphrase required to load an encrypted .hxe file")
            hx_path = path[:-1]  # strip the 'e'
            self.crypt.decrypt_file(path, hx_path, self.passphrase)
            self.fusion.load(hx_path)
            os.remove(hx_path)
        else:
            hx_path = path if path.endswith(".hx") else path + ".hx"
            self.fusion.load(hx_path)

        self._last_phi = self.fusion.recall_compact()
        self.tpi.force_record(0, self._last_phi)

    # ------------------------------------------------------------------
    # Stats
    # ------------------------------------------------------------------

    def stats(self):
        return {
            "steps_absorbed": self._step,
            "crystal_size_bytes": self.fusion.crystal.size_bytes(),
            "tpi_snapshots": self.tpi.num_snapshots(),
            "tpi_memory_bytes": self.tpi.memory_bytes(),
            "affect": self.affect_state(),
            "flags_collapsed": self.flags.num_collapsed(),
            "phase_velocity": self.phase_velocity(),
            "committed_versions": self.tracker.num_versions(),
            "encrypted": self.crypt is not None,
        }

    def __repr__(self):
        return (
            f"HelixMemory(steps={self._step}, "
            f"size={self.fusion.crystal.size_bytes()}B, "
            f"encrypted={self.crypt is not None})"
        )
