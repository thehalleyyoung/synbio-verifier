"""SBOL (Synthetic Biology Open Language) import module for BioProver.

Parses SBOL v2/v3 XML/RDF files into BioProver's internal BioModel
representation using only the standard-library xml.etree.ElementTree
parser.

Extracts ComponentDefinitions (genes, promoters, RBSs, terminators),
circuit topology from functional compositions, and maps common genetic
parts to standard kinetic models with default parameters.

Typical usage::

    model = parse_sbol_file("inverter.sbol")
    # or
    importer = SBOLImporter()
    model = importer.import_file("inverter.sbol")
    for w in importer.warnings:
        print(w)
"""

from __future__ import annotations

import logging
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# SBOL namespace constants
# ---------------------------------------------------------------------------

SBOL2_NS = "http://sbols.org/v2#"
SBOL3_NS = "http://sbols.org/v3#"
RDF_NS = "http://www.w3.org/1999/02/22-rdf-syntax-ns#"
DCTERMS_NS = "http://purl.org/dc/terms/"

# SBO (Systems Biology Ontology) role URIs for genetic parts
SO_PROMOTER = "http://identifiers.org/so/SO:0000167"
SO_RBS = "http://identifiers.org/so/SO:0000139"
SO_CDS = "http://identifiers.org/so/SO:0000316"
SO_TERMINATOR = "http://identifiers.org/so/SO:0000141"
SO_GENE = "http://identifiers.org/so/SO:0000704"
SO_ENGINEERED_REGION = "http://identifiers.org/so/SO:0000804"

# SBO interaction type URIs
SBO_INHIBITION = "http://identifiers.org/biomodels.sbo/SBO:0000169"
SBO_STIMULATION = "http://identifiers.org/biomodels.sbo/SBO:0000170"
SBO_GENETIC_PRODUCTION = "http://identifiers.org/biomodels.sbo/SBO:0000589"
SBO_DEGRADATION = "http://identifiers.org/biomodels.sbo/SBO:0000179"

# Participation role URIs
SBO_INHIBITOR = "http://identifiers.org/biomodels.sbo/SBO:0000020"
SBO_STIMULATOR = "http://identifiers.org/biomodels.sbo/SBO:0000459"
SBO_TEMPLATE = "http://identifiers.org/biomodels.sbo/SBO:0000645"
SBO_PRODUCT = "http://identifiers.org/biomodels.sbo/SBO:0000011"

# Role classification for kinetic model mapping
_ROLE_TO_PART_TYPE: Dict[str, str] = {
    SO_PROMOTER: "promoter",
    SO_RBS: "rbs",
    SO_CDS: "cds",
    SO_TERMINATOR: "terminator",
    SO_GENE: "cds",
    SO_ENGINEERED_REGION: "engineered_region",
}


# ---------------------------------------------------------------------------
# ImportWarning dataclass
# ---------------------------------------------------------------------------

@dataclass
class SBOLImportWarning:
    """Diagnostic emitted during SBOL import."""

    message: str
    severity: str = "warning"
    element: Optional[str] = None

    def __str__(self) -> str:
        prefix = f"[{self.severity.upper()}]"
        loc = f" ({self.element})" if self.element else ""
        return f"{prefix}{loc} {self.message}"


# ---------------------------------------------------------------------------
# Default kinetic parameters for genetic parts
# ---------------------------------------------------------------------------

# Default kinetic parameters used when generating a BioModel from SBOL.
# These are typical literature values for E. coli genetic circuits.
DEFAULT_PARAMS: Dict[str, Dict[str, float]] = {
    "promoter_constitutive": {
        "transcription_rate": 0.5,  # nM/min
    },
    "promoter_activated": {
        "Vmax": 10.0,  # nM/min
        "K": 2.0,      # nM (half-activation)
        "n": 2.0,      # Hill coefficient
    },
    "promoter_repressed": {
        "Vmax": 10.0,  # nM/min
        "K": 2.0,      # nM (half-repression)
        "n": 2.0,      # Hill coefficient
    },
    "rbs": {
        "translation_rate": 0.1,  # 1/min
    },
    "degradation": {
        "degradation_rate": 0.01,  # 1/min (protein)
        "mrna_degradation_rate": 0.1,  # 1/min (mRNA)
    },
}


# ---------------------------------------------------------------------------
# SBOLImporter
# ---------------------------------------------------------------------------

class SBOLImporter:
    """Import an SBOL v2/v3 document into BioProver's BioModel.

    After calling :meth:`import_file` or :meth:`import_string`, inspect
    :attr:`warnings` for any diagnostics produced during parsing.
    """

    def __init__(self) -> None:
        self.warnings: List[SBOLImportWarning] = []
        self._ns: Dict[str, str] = {}

    # -- public entry points -----------------------------------------------

    def import_file(self, filepath: str) -> "BioModel":
        """Parse an SBOL file from *filepath* and return a BioModel.

        Raises
        ------
        FileNotFoundError
            If the file does not exist.
        ET.ParseError
            If the XML is malformed.
        """
        path = Path(filepath)
        if not path.exists():
            raise FileNotFoundError(f"SBOL file not found: {filepath}")
        tree = ET.parse(str(path))
        root = tree.getroot()
        logger.info("Parsing SBOL file: %s", filepath)
        return self._parse_sbol(root, name=path.stem)

    def import_string(self, xml_string: str, name: str = "sbol_model") -> "BioModel":
        """Parse an SBOL document from an in-memory string.

        Raises
        ------
        ET.ParseError
            If the XML is malformed.
        """
        root = ET.fromstring(xml_string)
        logger.info("Parsing SBOL from string (%d chars)", len(xml_string))
        return self._parse_sbol(root, name=name)

    # -- private implementation --------------------------------------------

    def _parse_sbol(self, root: ET.Element, name: str = "sbol_model") -> "BioModel":
        """Main parsing pipeline: root XML element → BioModel."""
        from bioprover.models.bio_model import BioModel
        from bioprover.models.species import Species, SpeciesType
        from bioprover.models.reactions import (
            Reaction,
            StoichiometryEntry,
            HillActivation,
            HillRepression,
            LinearDegradation,
            ConstitutiveProduction,
            MassAction,
        )
        from bioprover.models.parameters import Parameter

        self._setup_namespaces(root)

        # Extract component definitions
        comp_defs = self._parse_component_definitions(root)
        # Extract interactions (from ModuleDefinitions)
        interactions = self._parse_interactions(root)

        model = BioModel(name=name)

        # Identify proteins produced by CDS parts
        proteins: Dict[str, str] = {}  # cds_id -> protein_species_name
        promoters: Dict[str, Dict[str, Any]] = {}  # promoter_id -> info

        for cd_id, cd_info in comp_defs.items():
            part_type = cd_info.get("part_type", "unknown")

            if part_type == "cds":
                protein_name = cd_info.get("name", cd_id) + "_protein"
                # Sanitize name for use as species
                protein_name = _sanitize_name(protein_name)
                proteins[cd_id] = protein_name
                sp = Species(
                    name=protein_name,
                    initial_concentration=0.0,
                    species_type=SpeciesType.PROTEIN,
                )
                try:
                    model.add_species(sp)
                except ValueError:
                    pass  # already exists

            elif part_type == "promoter":
                prom_name = cd_info.get("name", cd_id)
                promoters[cd_id] = {
                    "name": prom_name,
                    "display_id": cd_info.get("display_id", cd_id),
                }

        # Build interactions from SBOL interaction data
        if interactions:
            self._build_reactions_from_interactions(
                model, interactions, comp_defs, proteins
            )
        else:
            # Infer topology from component ordering in functional compositions
            self._build_reactions_from_topology(
                model, comp_defs, proteins, promoters
            )

        # Add degradation reactions for all protein species
        for protein_name in proteins.values():
            deg_rate = DEFAULT_PARAMS["degradation"]["degradation_rate"]
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

        # Add default parameters to the model
        for param_group_name, params in DEFAULT_PARAMS.items():
            for p_name, p_val in params.items():
                full_name = f"{param_group_name}_{p_name}"
                model.add_parameter(Parameter(name=full_name, value=p_val))

        logger.info(
            "SBOL import complete: %d species, %d reactions, %d warnings",
            model.num_species,
            model.num_reactions,
            len(self.warnings),
        )
        return model

    def _setup_namespaces(self, root: ET.Element) -> None:
        """Detect SBOL version from root element namespaces."""
        tag = root.tag
        self._ns = {
            "rdf": RDF_NS,
            "dcterms": DCTERMS_NS,
        }

        # Check for SBOL2 or SBOL3 namespace in root tag or attributes
        if SBOL2_NS.rstrip("#") in tag or SBOL2_NS.rstrip("#") in str(
            ET.tostring(root, encoding="unicode")[:500]
        ):
            self._ns["sbol"] = SBOL2_NS
            logger.debug("Detected SBOL v2")
        elif SBOL3_NS.rstrip("#") in tag or SBOL3_NS.rstrip("#") in str(
            ET.tostring(root, encoding="unicode")[:500]
        ):
            self._ns["sbol"] = SBOL3_NS
            logger.debug("Detected SBOL v3")
        else:
            self._ns["sbol"] = SBOL2_NS
            self.warnings.append(
                SBOLImportWarning(
                    message="Could not detect SBOL version; assuming v2",
                    severity="warning",
                    element="rdf:RDF",
                )
            )

    def _parse_component_definitions(
        self, root: ET.Element
    ) -> Dict[str, Dict[str, Any]]:
        """Extract ComponentDefinition elements from the SBOL document."""
        comp_defs: Dict[str, Dict[str, Any]] = {}
        sbol_ns = self._ns.get("sbol", SBOL2_NS)

        # Try SBOL2 ComponentDefinition
        for elem in root.iter(f"{{{sbol_ns}}}ComponentDefinition"):
            cd_id = elem.get(f"{{{RDF_NS}}}about", "")
            display_id = self._get_text(elem, f"{{{sbol_ns}}}displayId")
            name = self._get_text(elem, f"{{{DCTERMS_NS}}}title")
            if not name:
                name = display_id or cd_id.split("/")[-1]

            # Determine part type from roles
            part_type = "unknown"
            for role_elem in elem.iter(f"{{{sbol_ns}}}role"):
                role_uri = role_elem.get(f"{{{RDF_NS}}}resource", "")
                if role_uri in _ROLE_TO_PART_TYPE:
                    part_type = _ROLE_TO_PART_TYPE[role_uri]
                    break

            # Collect sub-components
            components: List[str] = []
            for comp_elem in elem.iter(f"{{{sbol_ns}}}component"):
                comp_def_ref = ""
                for def_elem in comp_elem.iter(f"{{{sbol_ns}}}definition"):
                    comp_def_ref = def_elem.get(
                        f"{{{RDF_NS}}}resource", ""
                    )
                if comp_def_ref:
                    components.append(comp_def_ref)

            # Collect sequence constraints for ordering
            sequence_constraints: List[Tuple[str, str]] = []
            for sc_elem in elem.iter(f"{{{sbol_ns}}}sequenceConstraint"):
                subject = ""
                obj = ""
                for s in sc_elem.iter(f"{{{sbol_ns}}}subject"):
                    subject = s.get(f"{{{RDF_NS}}}resource", "")
                for o in sc_elem.iter(f"{{{sbol_ns}}}object"):
                    obj = o.get(f"{{{RDF_NS}}}resource", "")
                if subject and obj:
                    sequence_constraints.append((subject, obj))

            comp_defs[cd_id] = {
                "name": name,
                "display_id": display_id or cd_id.split("/")[-1],
                "part_type": part_type,
                "components": components,
                "sequence_constraints": sequence_constraints,
            }

        # Try SBOL3 Component (v3 renamed ComponentDefinition to Component)
        for elem in root.iter(f"{{{sbol_ns}}}Component"):
            cd_id = elem.get(f"{{{RDF_NS}}}about", "")
            if cd_id in comp_defs:
                continue
            display_id = self._get_text(elem, f"{{{sbol_ns}}}displayId")
            name = self._get_text(elem, f"{{{sbol_ns}}}name")
            if not name:
                name = display_id or cd_id.split("/")[-1]

            part_type = "unknown"
            for role_elem in elem.iter(f"{{{sbol_ns}}}role"):
                role_uri = role_elem.get(f"{{{RDF_NS}}}resource", "")
                if role_uri in _ROLE_TO_PART_TYPE:
                    part_type = _ROLE_TO_PART_TYPE[role_uri]
                    break

            comp_defs[cd_id] = {
                "name": name,
                "display_id": display_id or cd_id.split("/")[-1],
                "part_type": part_type,
                "components": [],
                "sequence_constraints": [],
            }

        if not comp_defs:
            self.warnings.append(
                SBOLImportWarning(
                    message="No ComponentDefinition elements found",
                    severity="warning",
                )
            )

        return comp_defs

    def _parse_interactions(
        self, root: ET.Element
    ) -> List[Dict[str, Any]]:
        """Extract Interaction elements from ModuleDefinitions."""
        interactions: List[Dict[str, Any]] = []
        sbol_ns = self._ns.get("sbol", SBOL2_NS)

        for mod_def in root.iter(f"{{{sbol_ns}}}ModuleDefinition"):
            for interaction in mod_def.iter(f"{{{sbol_ns}}}interaction"):
                int_info: Dict[str, Any] = {"type": "", "participations": []}

                # Get interaction type
                for type_elem in interaction.iter(f"{{{sbol_ns}}}type"):
                    int_info["type"] = type_elem.get(
                        f"{{{RDF_NS}}}resource", ""
                    )

                # Get participations
                for part in interaction.iter(f"{{{sbol_ns}}}participation"):
                    p_info: Dict[str, str] = {"role": "", "participant": ""}
                    for role_elem in part.iter(f"{{{sbol_ns}}}role"):
                        p_info["role"] = role_elem.get(
                            f"{{{RDF_NS}}}resource", ""
                        )
                    for participant in part.iter(f"{{{sbol_ns}}}participant"):
                        p_info["participant"] = participant.get(
                            f"{{{RDF_NS}}}resource", ""
                        )
                    int_info["participations"].append(p_info)

                if int_info["type"]:
                    interactions.append(int_info)

        return interactions

    def _build_reactions_from_interactions(
        self,
        model: "BioModel",
        interactions: List[Dict[str, Any]],
        comp_defs: Dict[str, Dict[str, Any]],
        proteins: Dict[str, str],
    ) -> None:
        """Build BioModel reactions from SBOL interaction data."""
        from bioprover.models.reactions import (
            Reaction,
            StoichiometryEntry,
            HillActivation,
            HillRepression,
            ConstitutiveProduction,
        )
        from bioprover.models.species import Species, SpeciesType

        for idx, interaction in enumerate(interactions):
            int_type = interaction["type"]
            participants = interaction["participations"]

            if int_type == SBO_INHIBITION:
                inhibitor = None
                product = None
                for p in participants:
                    role = p["role"]
                    part_uri = p["participant"]
                    if role == SBO_INHIBITOR:
                        inhibitor = self._resolve_protein_name(
                            part_uri, comp_defs, proteins, model
                        )
                    elif role == SBO_PRODUCT:
                        product = self._resolve_protein_name(
                            part_uri, comp_defs, proteins, model
                        )

                if inhibitor and product:
                    params = DEFAULT_PARAMS["promoter_repressed"]
                    hill = HillRepression(
                        Vmax=params["Vmax"], K=params["K"], n=params["n"]
                    )
                    hill.repressor_name = inhibitor
                    rxn = Reaction(
                        name=f"repression_{idx}_{product}",
                        reactants=[],
                        products=[StoichiometryEntry(product, 1)],
                        kinetic_law=hill,
                        modifiers=[inhibitor],
                    )
                    try:
                        model.add_reaction(rxn)
                    except ValueError:
                        pass

            elif int_type == SBO_STIMULATION:
                stimulator = None
                product = None
                for p in participants:
                    role = p["role"]
                    part_uri = p["participant"]
                    if role == SBO_STIMULATOR:
                        stimulator = self._resolve_protein_name(
                            part_uri, comp_defs, proteins, model
                        )
                    elif role == SBO_PRODUCT:
                        product = self._resolve_protein_name(
                            part_uri, comp_defs, proteins, model
                        )

                if stimulator and product:
                    params = DEFAULT_PARAMS["promoter_activated"]
                    hill = HillActivation(
                        Vmax=params["Vmax"], K=params["K"], n=params["n"]
                    )
                    hill.activator_name = stimulator
                    rxn = Reaction(
                        name=f"activation_{idx}_{product}",
                        reactants=[],
                        products=[StoichiometryEntry(product, 1)],
                        kinetic_law=hill,
                        modifiers=[stimulator],
                    )
                    try:
                        model.add_reaction(rxn)
                    except ValueError:
                        pass

            elif int_type == SBO_GENETIC_PRODUCTION:
                product = None
                for p in participants:
                    if p["role"] == SBO_PRODUCT:
                        product = self._resolve_protein_name(
                            p["participant"], comp_defs, proteins, model
                        )
                if product:
                    rate = DEFAULT_PARAMS["promoter_constitutive"][
                        "transcription_rate"
                    ]
                    rxn = Reaction(
                        name=f"production_{idx}_{product}",
                        reactants=[],
                        products=[StoichiometryEntry(product, 1)],
                        kinetic_law=ConstitutiveProduction(rate=rate),
                    )
                    try:
                        model.add_reaction(rxn)
                    except ValueError:
                        pass

    def _build_reactions_from_topology(
        self,
        model: "BioModel",
        comp_defs: Dict[str, Dict[str, Any]],
        proteins: Dict[str, str],
        promoters: Dict[str, Dict[str, Any]],
    ) -> None:
        """Infer reactions from component ordering within compositions.

        When no explicit interactions are defined, we examine the ordered
        sequence of parts within composite ComponentDefinitions:
        promoter → RBS → CDS → terminator, and assign constitutive
        production kinetics.
        """
        from bioprover.models.reactions import (
            Reaction,
            StoichiometryEntry,
            ConstitutiveProduction,
        )

        # Find composite parts (those with sub-components)
        for cd_id, cd_info in comp_defs.items():
            if not cd_info["components"]:
                continue

            # Walk sub-components in order and pair promoter→CDS
            current_promoter = None
            for comp_ref in cd_info["components"]:
                sub = comp_defs.get(comp_ref, {})
                sub_type = sub.get("part_type", "unknown")

                if sub_type == "promoter":
                    current_promoter = comp_ref
                elif sub_type == "cds" and comp_ref in proteins:
                    protein_name = proteins[comp_ref]
                    rate = DEFAULT_PARAMS["promoter_constitutive"][
                        "transcription_rate"
                    ]
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
                    current_promoter = None

    def _resolve_protein_name(
        self,
        part_uri: str,
        comp_defs: Dict[str, Dict[str, Any]],
        proteins: Dict[str, str],
        model: "BioModel",
    ) -> Optional[str]:
        """Resolve a participant URI to a protein species name.

        If the URI references a FunctionalComponent, resolve through to
        the underlying ComponentDefinition's protein.
        """
        from bioprover.models.species import Species, SpeciesType

        # Direct match in proteins dict
        if part_uri in proteins:
            return proteins[part_uri]

        # Check if it's a component definition with a CDS
        if part_uri in comp_defs:
            cd = comp_defs[part_uri]
            if cd["part_type"] == "cds":
                pname = _sanitize_name(cd.get("name", part_uri) + "_protein")
                if pname not in proteins.values():
                    proteins[part_uri] = pname
                    try:
                        model.add_species(
                            Species(
                                name=pname,
                                initial_concentration=0.0,
                                species_type=SpeciesType.PROTEIN,
                            )
                        )
                    except ValueError:
                        pass
                return pname

        # Fallback: create a species from the URI fragment
        fragment = part_uri.split("/")[-1]
        pname = _sanitize_name(fragment)
        if pname not in [sp.name for sp in model.species]:
            try:
                model.add_species(
                    Species(
                        name=pname,
                        initial_concentration=0.0,
                        species_type=SpeciesType.PROTEIN,
                    )
                )
            except ValueError:
                pass
        return pname

    @staticmethod
    def _get_text(elem: ET.Element, tag: str) -> Optional[str]:
        """Get text content of a child element, or None."""
        child = elem.find(tag)
        if child is not None and child.text:
            return child.text.strip()
        return None


# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------

def _sanitize_name(name: str) -> str:
    """Sanitize a string for use as a BioProver species/reaction name.

    Replaces non-alphanumeric characters with underscores and strips
    leading/trailing underscores.
    """
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

def parse_sbol_file(filepath: str) -> "BioModel":
    """Parse an SBOL file and return a BioModel.

    Convenience wrapper around :class:`SBOLImporter`.

    Parameters
    ----------
    filepath : str
        Path to the SBOL XML/RDF file.

    Returns
    -------
    BioModel
        The imported biological model with default kinetic parameters.
    """
    importer = SBOLImporter()
    return importer.import_file(filepath)


def parse_sbol_string(xml_string: str, name: str = "sbol_model") -> "BioModel":
    """Parse an SBOL document from a string and return a BioModel.

    Convenience wrapper around :class:`SBOLImporter`.

    Parameters
    ----------
    xml_string : str
        SBOL XML/RDF content.
    name : str
        Name for the resulting model.

    Returns
    -------
    BioModel
        The imported biological model with default kinetic parameters.
    """
    importer = SBOLImporter()
    return importer.import_string(xml_string, name=name)
