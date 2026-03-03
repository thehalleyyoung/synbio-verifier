"""GenBank flat file import module for BioProver.

Parses GenBank (.gb/.gbk/.genbank) annotation files and extracts
FEATURES (CDS, promoter, terminator, RBS, regulatory) to build a
BioProver BioModel with default kinetic parameters.

Only the feature annotations are used — the raw DNA sequence is not
parsed.

Typical usage::

    model = parse_genbank_file("toggle_switch.gb")
    # or
    importer = GenBankImporter()
    model = importer.import_file("toggle_switch.gb")
    for w in importer.warnings:
        print(w)
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# Re-use the same default kinetic parameters as the SBOL importer.
from bioprover.models.sbol_import import DEFAULT_PARAMS

# ---------------------------------------------------------------------------
# GenBankImportWarning
# ---------------------------------------------------------------------------

@dataclass
class GenBankImportWarning:
    """Diagnostic emitted during GenBank import."""

    message: str
    severity: str = "warning"
    element: Optional[str] = None

    def __str__(self) -> str:
        prefix = f"[{self.severity.upper()}]"
        loc = f" ({self.element})" if self.element else ""
        return f"{prefix}{loc} {self.message}"


# ---------------------------------------------------------------------------
# Feature dataclass
# ---------------------------------------------------------------------------

@dataclass
class GenBankFeature:
    """A single parsed FEATURES table entry."""

    feature_type: str  # e.g. "CDS", "promoter", "terminator", …
    location: str = ""
    qualifiers: Dict[str, str] = field(default_factory=dict)

    @property
    def gene_name(self) -> Optional[str]:
        """Best-effort human-readable name for this feature."""
        for key in ("gene", "product", "label", "note"):
            if key in self.qualifiers:
                return self.qualifiers[key]
        return None


# ---------------------------------------------------------------------------
# GenBankImporter
# ---------------------------------------------------------------------------

# Feature types recognised for kinetic-model mapping (case-insensitive).
_FEATURE_TYPE_MAP = {
    "cds": "cds",
    "promoter": "promoter",
    "terminator": "terminator",
    "rbs": "rbs",
    "regulatory": "regulatory",
    "misc_feature": "misc",
}


class GenBankImporter:
    """Import a GenBank flat file into BioProver's BioModel.

    After calling :meth:`import_file` or :meth:`import_string`, inspect
    :attr:`warnings` for any diagnostics produced during parsing.
    """

    def __init__(self) -> None:
        self.warnings: List[GenBankImportWarning] = []

    # -- public entry points -----------------------------------------------

    def import_file(self, filepath: str) -> "BioModel":
        """Parse a GenBank file and return a BioModel.

        Raises
        ------
        FileNotFoundError
            If the file does not exist.
        """
        path = Path(filepath)
        if not path.exists():
            raise FileNotFoundError(f"GenBank file not found: {filepath}")
        text = path.read_text(encoding="utf-8", errors="replace")
        logger.info("Parsing GenBank file: %s", filepath)
        return self._build_model(self._parse_features(text), name=path.stem)

    def import_string(self, text: str, name: str = "genbank_model") -> "BioModel":
        """Parse a GenBank document from an in-memory string."""
        logger.info("Parsing GenBank from string (%d chars)", len(text))
        return self._build_model(self._parse_features(text), name=name)

    # -- feature parsing ---------------------------------------------------

    def _parse_features(self, text: str) -> List[GenBankFeature]:
        """Extract features from the FEATURES table of a GenBank file."""
        features: List[GenBankFeature] = []

        # Locate the FEATURES block (ends at ORIGIN or the end of file).
        feat_match = re.search(
            r"^FEATURES\s+Location/Qualifiers\s*$",
            text,
            re.MULTILINE,
        )
        if feat_match is None:
            self.warnings.append(
                GenBankImportWarning("No FEATURES section found", element="FEATURES")
            )
            return features

        start = feat_match.end()
        end_match = re.search(r"^(ORIGIN|//)\b", text[start:], re.MULTILINE)
        feat_block = text[start : start + end_match.start()] if end_match else text[start:]

        current: Optional[GenBankFeature] = None
        current_qual_key: Optional[str] = None

        for line in feat_block.split("\n"):
            # Feature key line: 5-char indent then feature type
            m_feat = re.match(r"^     (\S+)\s+(.*)", line)
            # Qualifier line: 21-char indent then /key="value"
            m_qual = re.match(r"^\s{21}/(\w+)=\"?([^\"]*)\"?\s*$", line)
            # Continuation line (21-char indent, no /)
            m_cont = re.match(r"^\s{21}([^/].*)", line)

            if m_feat:
                if current is not None:
                    features.append(current)
                current = GenBankFeature(
                    feature_type=m_feat.group(1),
                    location=m_feat.group(2).strip(),
                )
                current_qual_key = None
            elif m_qual and current is not None:
                key = m_qual.group(1)
                val = m_qual.group(2).rstrip('"')
                current.qualifiers[key] = val
                current_qual_key = key
            elif m_cont and current is not None and current_qual_key is not None:
                # Append continuation to the last qualifier value.
                current.qualifiers[current_qual_key] += " " + m_cont.group(1).strip().rstrip('"')

        if current is not None:
            features.append(current)

        return features

    # -- model building ----------------------------------------------------

    def _build_model(
        self, features: List[GenBankFeature], name: str
    ) -> "BioModel":
        """Convert parsed GenBank features into a BioModel."""
        from bioprover.models.bio_model import BioModel
        from bioprover.models.species import Species, SpeciesType
        from bioprover.models.reactions import (
            Reaction,
            StoichiometryEntry,
            HillActivation,
            HillRepression,
            LinearDegradation,
            ConstitutiveProduction,
        )
        from bioprover.models.parameters import Parameter

        model = BioModel(name=name)

        proteins: Dict[str, str] = {}  # feature_name -> species name
        pending_regulation: List[GenBankFeature] = []
        last_promoter: Optional[str] = None

        # First pass: create species for CDS features and collect topology.
        for feat in features:
            ft = feat.feature_type.lower()
            mapped = _FEATURE_TYPE_MAP.get(ft)

            if mapped == "cds":
                gene = feat.gene_name or feat.feature_type
                protein_name = _sanitize_name(gene) + "_protein"
                if protein_name in proteins.values():
                    continue
                proteins[gene] = protein_name
                try:
                    model.add_species(
                        Species(
                            name=protein_name,
                            initial_concentration=0.0,
                            species_type=SpeciesType.PROTEIN,
                        )
                    )
                except ValueError:
                    pass

            elif mapped == "promoter":
                last_promoter = feat.gene_name or feat.feature_type

            elif mapped == "regulatory":
                pending_regulation.append(feat)

        # Second pass: build reactions.
        # Walk features in order to pair promoter→CDS (same as SBOL topology).
        last_promoter = None
        regulated_cds: Dict[str, List[GenBankFeature]] = {}

        for feat in features:
            ft = feat.feature_type.lower()
            mapped = _FEATURE_TYPE_MAP.get(ft)

            if mapped == "promoter":
                last_promoter = feat.gene_name or feat.feature_type

            elif mapped == "cds":
                gene = feat.gene_name or feat.feature_type
                protein_name = proteins.get(gene)
                if protein_name is None:
                    continue

                # Check if any regulatory feature targets this CDS.
                regulators = [
                    r for r in pending_regulation
                    if self._regulation_targets(r, gene)
                ]

                if regulators:
                    for reg in regulators:
                        reg_type = self._classify_regulation(reg)
                        regulator_gene = reg.qualifiers.get(
                            "gene", reg.qualifiers.get("note", "")
                        )
                        regulator_protein = proteins.get(regulator_gene)
                        if regulator_protein is None:
                            # Create the regulator species if needed.
                            regulator_protein = _sanitize_name(regulator_gene) + "_protein"
                            if regulator_protein not in proteins.values():
                                proteins[regulator_gene] = regulator_protein
                                try:
                                    model.add_species(
                                        Species(
                                            name=regulator_protein,
                                            initial_concentration=0.0,
                                            species_type=SpeciesType.PROTEIN,
                                        )
                                    )
                                except ValueError:
                                    pass

                        if reg_type == "repression":
                            params = DEFAULT_PARAMS["promoter_repressed"]
                            hill = HillRepression(
                                Vmax=params["Vmax"], K=params["K"], n=params["n"]
                            )
                            hill.repressor_name = regulator_protein
                            rxn = Reaction(
                                name=f"repression_{protein_name}",
                                reactants=[],
                                products=[StoichiometryEntry(protein_name, 1)],
                                kinetic_law=hill,
                                modifiers=[regulator_protein],
                            )
                        else:
                            params = DEFAULT_PARAMS["promoter_activated"]
                            hill = HillActivation(
                                Vmax=params["Vmax"], K=params["K"], n=params["n"]
                            )
                            hill.activator_name = regulator_protein
                            rxn = Reaction(
                                name=f"activation_{protein_name}",
                                reactants=[],
                                products=[StoichiometryEntry(protein_name, 1)],
                                kinetic_law=hill,
                                modifiers=[regulator_protein],
                            )
                        try:
                            model.add_reaction(rxn)
                        except ValueError:
                            pass
                else:
                    # Constitutive production.
                    rate = DEFAULT_PARAMS["promoter_constitutive"]["transcription_rate"]
                    rxn = Reaction(
                        name=f"constitutive_{protein_name}",
                        reactants=[],
                        products=[StoichiometryEntry(protein_name, 1)],
                        kinetic_law=ConstitutiveProduction(rate=rate),
                    )
                    try:
                        model.add_reaction(rxn)
                    except ValueError:
                        pass

                last_promoter = None

        # Degradation reactions for every protein.
        deg_rate = DEFAULT_PARAMS["degradation"]["degradation_rate"]
        for protein_name in proteins.values():
            deg_law = LinearDegradation(rate=deg_rate)
            deg_law.species_name = protein_name
            rxn = Reaction(
                name=f"{protein_name}_degradation",
                reactants=[StoichiometryEntry(protein_name, 1)],
                products=[],
                kinetic_law=deg_law,
            )
            try:
                model.add_reaction(rxn)
            except ValueError:
                pass

        # Add default parameters to the model.
        for param_group_name, params in DEFAULT_PARAMS.items():
            for p_name, p_val in params.items():
                full_name = f"{param_group_name}_{p_name}"
                model.add_parameter(Parameter(name=full_name, value=p_val))

        if model.num_species == 0:
            self.warnings.append(
                GenBankImportWarning(
                    "No CDS features found — model has no species",
                    severity="warning",
                )
            )

        logger.info(
            "GenBank import complete: %d species, %d reactions, %d warnings",
            model.num_species,
            model.num_reactions,
            len(self.warnings),
        )
        return model

    # -- helpers -----------------------------------------------------------

    @staticmethod
    def _regulation_targets(reg_feat: GenBankFeature, gene: str) -> bool:
        """Heuristic: does *reg_feat* regulate *gene*?"""
        note = reg_feat.qualifiers.get("note", "").lower()
        product = reg_feat.qualifiers.get("product", "").lower()
        bound = reg_feat.qualifiers.get("bound_moiety", "").lower()
        gene_lower = gene.lower()
        return (
            gene_lower in note
            or gene_lower in product
            or gene_lower in bound
        )

    @staticmethod
    def _classify_regulation(reg_feat: GenBankFeature) -> str:
        """Return ``'repression'`` or ``'activation'`` for a regulatory feature."""
        func = reg_feat.qualifiers.get("regulatory_class", "").lower()
        note = reg_feat.qualifiers.get("note", "").lower()
        for kw in ("repress", "inhibit", "negati"):
            if kw in func or kw in note:
                return "repression"
        return "activation"


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------

def _sanitize_name(name: str) -> str:
    """Sanitize a string for use as a BioProver species/reaction name."""
    result = ""
    for ch in name:
        if ch.isalnum() or ch == "_":
            result += ch
        else:
            result += "_"
    return result.strip("_") or "unnamed"


# ---------------------------------------------------------------------------
# Convenience functions
# ---------------------------------------------------------------------------

def parse_genbank_file(filepath: str) -> "BioModel":
    """Parse a GenBank file and return a BioModel.

    Convenience wrapper around :class:`GenBankImporter`.

    Parameters
    ----------
    filepath : str
        Path to the GenBank file (.gb / .gbk / .genbank).

    Returns
    -------
    BioModel
        The imported biological model with default kinetic parameters.
    """
    importer = GenBankImporter()
    return importer.import_file(filepath)


def parse_genbank_string(text: str, name: str = "genbank_model") -> "BioModel":
    """Parse a GenBank document from a string and return a BioModel.

    Convenience wrapper around :class:`GenBankImporter`.

    Parameters
    ----------
    text : str
        GenBank flat-file content.
    name : str
        Name for the resulting model.

    Returns
    -------
    BioModel
        The imported biological model with default kinetic parameters.
    """
    importer = GenBankImporter()
    return importer.import_string(text, name=name)
