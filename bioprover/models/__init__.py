"""
BioProver models module — core biological model representation layer.

Provides species, reactions, regulatory networks, parameter uncertainty,
model transformations, and SBML import capabilities.
"""

from .species import (
    Species,
    SpeciesType,
    BoundaryCondition,
    DatabaseReference,
    SpeciesMetadata,
    ConcentrationBounds,
)
from .reactions import (
    Reaction,
    KineticLaw,
    MassAction,
    HillActivation,
    HillRepression,
    MichaelisMenten,
    ConstitutiveProduction,
    LinearDegradation,
    DimerFormation,
    StoichiometryEntry,
    build_stoichiometry_matrix,
    compute_propensity_vector,
)
from .regulatory_network import (
    GeneRegulatoryNetwork,
    RegulatoryInteraction,
    InteractionType,
    InteractionSign,
    FeedbackLoop,
    LoopType,
    NetworkMotif,
    MotifType,
)
from .bio_model import BioModel, Compartment
from .parameters import (
    Parameter,
    ParameterSet,
    UncertaintyType,
    UncertaintyEnvelope,
)
from .transforms import (
    ModelTransform,
    TransformType,
    TransformRecord,
    TransformHistory,
    QSSATransform,
    TimeScaleSeparation,
    SpeciesLumping,
    ConservationReduction,
    Nondimensionalization,
    ModelReductionPipeline,
    revert_solution,
)
from .sbml_import import (
    SBMLImporter,
    GenericKineticLaw,
    parse_sbml_file,
    parse_sbml_string,
)
from .sbol_import import (
    SBOLImporter,
    parse_sbol_file,
    parse_sbol_string,
)
from .genbank_import import (
    GenBankImporter,
    parse_genbank_file,
    parse_genbank_string,
)

__all__ = [
    # species
    "Species",
    "SpeciesType",
    "BoundaryCondition",
    "DatabaseReference",
    "SpeciesMetadata",
    "ConcentrationBounds",
    # reactions
    "Reaction",
    "KineticLaw",
    "MassAction",
    "HillActivation",
    "HillRepression",
    "MichaelisMenten",
    "ConstitutiveProduction",
    "LinearDegradation",
    "DimerFormation",
    "StoichiometryEntry",
    "build_stoichiometry_matrix",
    "compute_propensity_vector",
    # regulatory network
    "GeneRegulatoryNetwork",
    "RegulatoryInteraction",
    "InteractionType",
    "InteractionSign",
    "FeedbackLoop",
    "LoopType",
    "NetworkMotif",
    "MotifType",
    # bio_model
    "BioModel",
    "Compartment",
    # parameters
    "Parameter",
    "ParameterSet",
    "UncertaintyType",
    "UncertaintyEnvelope",
    # transforms
    "ModelTransform",
    "TransformType",
    "TransformRecord",
    "TransformHistory",
    "QSSATransform",
    "TimeScaleSeparation",
    "SpeciesLumping",
    "ConservationReduction",
    "Nondimensionalization",
    "ModelReductionPipeline",
    "revert_solution",
    # sbml_import
    "SBMLImporter",
    "GenericKineticLaw",
    "parse_sbml_file",
    "parse_sbml_string",
    # sbol_import
    "SBOLImporter",
    "parse_sbol_file",
    "parse_sbol_string",
    # genbank_import
    "GenBankImporter",
    "parse_genbank_file",
    "parse_genbank_string",
]
