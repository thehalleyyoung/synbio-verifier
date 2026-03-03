"""Unit tests for SBOL import module."""

import os
import tempfile

import pytest

from bioprover.models.sbol_import import (
    SBOLImporter,
    parse_sbol_file,
    parse_sbol_string,
    _sanitize_name,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

INVERTER_SBOL = """\
<?xml version="1.0" encoding="UTF-8"?>
<rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#"
         xmlns:dcterms="http://purl.org/dc/terms/"
         xmlns:sbol="http://sbols.org/v2#">

  <sbol:ComponentDefinition rdf:about="https://example.org/cd/LacI">
    <sbol:displayId>LacI</sbol:displayId>
    <dcterms:title>LacI</dcterms:title>
    <sbol:type rdf:resource="http://www.biopax.org/release/biopax-level3.owl#Protein"/>
    <sbol:role rdf:resource="http://identifiers.org/so/SO:0000316"/>
  </sbol:ComponentDefinition>

  <sbol:ComponentDefinition rdf:about="https://example.org/cd/GFP_cds">
    <sbol:displayId>GFP_cds</sbol:displayId>
    <dcterms:title>GFP</dcterms:title>
    <sbol:type rdf:resource="http://www.biopax.org/release/biopax-level3.owl#DnaRegion"/>
    <sbol:role rdf:resource="http://identifiers.org/so/SO:0000316"/>
  </sbol:ComponentDefinition>

  <sbol:ComponentDefinition rdf:about="https://example.org/cd/pLac">
    <sbol:displayId>pLac</sbol:displayId>
    <dcterms:title>pLac</dcterms:title>
    <sbol:type rdf:resource="http://www.biopax.org/release/biopax-level3.owl#DnaRegion"/>
    <sbol:role rdf:resource="http://identifiers.org/so/SO:0000167"/>
  </sbol:ComponentDefinition>

  <sbol:ModuleDefinition rdf:about="https://example.org/md/inverter">
    <sbol:displayId>inverter</sbol:displayId>
    <dcterms:title>Genetic Inverter</dcterms:title>

    <sbol:interaction>
      <sbol:Interaction rdf:about="https://example.org/md/inverter/int/repression">
        <sbol:type rdf:resource="http://identifiers.org/biomodels.sbo/SBO:0000169"/>
        <sbol:participation>
          <sbol:Participation>
            <sbol:role rdf:resource="http://identifiers.org/biomodels.sbo/SBO:0000020"/>
            <sbol:participant rdf:resource="https://example.org/cd/LacI"/>
          </sbol:Participation>
        </sbol:participation>
        <sbol:participation>
          <sbol:Participation>
            <sbol:role rdf:resource="http://identifiers.org/biomodels.sbo/SBO:0000011"/>
            <sbol:participant rdf:resource="https://example.org/cd/GFP_cds"/>
          </sbol:Participation>
        </sbol:participation>
      </sbol:Interaction>
    </sbol:interaction>
  </sbol:ModuleDefinition>
</rdf:RDF>
"""

MINIMAL_SBOL = """\
<?xml version="1.0" encoding="UTF-8"?>
<rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#"
         xmlns:dcterms="http://purl.org/dc/terms/"
         xmlns:sbol="http://sbols.org/v2#">

  <sbol:ComponentDefinition rdf:about="https://example.org/cd/GeneA">
    <sbol:displayId>GeneA</sbol:displayId>
    <dcterms:title>GeneA</dcterms:title>
    <sbol:role rdf:resource="http://identifiers.org/so/SO:0000316"/>
  </sbol:ComponentDefinition>
</rdf:RDF>
"""


ACTIVATION_SBOL = """\
<?xml version="1.0" encoding="UTF-8"?>
<rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#"
         xmlns:dcterms="http://purl.org/dc/terms/"
         xmlns:sbol="http://sbols.org/v2#">

  <sbol:ComponentDefinition rdf:about="https://example.org/cd/AraC">
    <sbol:displayId>AraC</sbol:displayId>
    <dcterms:title>AraC</dcterms:title>
    <sbol:role rdf:resource="http://identifiers.org/so/SO:0000316"/>
  </sbol:ComponentDefinition>

  <sbol:ComponentDefinition rdf:about="https://example.org/cd/GFP_cds">
    <sbol:displayId>GFP_cds</sbol:displayId>
    <dcterms:title>GFP</dcterms:title>
    <sbol:role rdf:resource="http://identifiers.org/so/SO:0000316"/>
  </sbol:ComponentDefinition>

  <sbol:ModuleDefinition rdf:about="https://example.org/md/activator">
    <sbol:displayId>activator</sbol:displayId>
    <sbol:interaction>
      <sbol:Interaction>
        <sbol:type rdf:resource="http://identifiers.org/biomodels.sbo/SBO:0000170"/>
        <sbol:participation>
          <sbol:Participation>
            <sbol:role rdf:resource="http://identifiers.org/biomodels.sbo/SBO:0000459"/>
            <sbol:participant rdf:resource="https://example.org/cd/AraC"/>
          </sbol:Participation>
        </sbol:participation>
        <sbol:participation>
          <sbol:Participation>
            <sbol:role rdf:resource="http://identifiers.org/biomodels.sbo/SBO:0000011"/>
            <sbol:participant rdf:resource="https://example.org/cd/GFP_cds"/>
          </sbol:Participation>
        </sbol:participation>
      </sbol:Interaction>
    </sbol:interaction>
  </sbol:ModuleDefinition>
</rdf:RDF>
"""


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestSBOLImporter:
    """Tests for the SBOLImporter class."""

    def test_parse_inverter_from_string(self):
        """Import an inverter circuit from SBOL string."""
        importer = SBOLImporter()
        model = importer.import_string(INVERTER_SBOL, name="inverter")

        assert model.name == "inverter"
        assert model.num_species >= 2  # LacI_protein and GFP_protein
        assert model.num_reactions >= 1  # at least repression reaction

    def test_inverter_has_repression_reaction(self):
        """The inverter should have a Hill repression reaction."""
        from bioprover.models.reactions import HillRepression

        model = parse_sbol_string(INVERTER_SBOL, name="inverter")

        hill_reactions = [
            r for r in model.reactions
            if isinstance(r.kinetic_law, HillRepression)
        ]
        assert len(hill_reactions) >= 1

    def test_inverter_has_degradation(self):
        """All protein species should have degradation reactions."""
        from bioprover.models.reactions import LinearDegradation

        model = parse_sbol_string(INVERTER_SBOL, name="inverter")

        deg_reactions = [
            r for r in model.reactions
            if isinstance(r.kinetic_law, LinearDegradation)
        ]
        assert len(deg_reactions) >= 1

    def test_parse_minimal_sbol(self):
        """A minimal SBOL with one CDS produces a model with one protein."""
        model = parse_sbol_string(MINIMAL_SBOL, name="minimal")

        assert model.num_species >= 1
        species_names = model.species_names
        assert any("GeneA" in n for n in species_names)

    def test_activation_interaction(self):
        """An activation interaction creates a HillActivation reaction."""
        from bioprover.models.reactions import HillActivation

        model = parse_sbol_string(ACTIVATION_SBOL, name="activator")

        hill_reactions = [
            r for r in model.reactions
            if isinstance(r.kinetic_law, HillActivation)
        ]
        assert len(hill_reactions) >= 1

    def test_import_file(self, tmp_path):
        """Test importing from a file."""
        sbol_file = tmp_path / "test.sbol"
        sbol_file.write_text(INVERTER_SBOL, encoding="utf-8")

        model = parse_sbol_file(str(sbol_file))
        assert model.num_species >= 2

    def test_import_file_not_found(self):
        """FileNotFoundError for missing file."""
        with pytest.raises(FileNotFoundError):
            parse_sbol_file("/nonexistent/path/model.sbol")

    def test_model_is_simulatable(self):
        """The generated model should be simulatable."""
        model = parse_sbol_string(INVERTER_SBOL, name="inverter")

        # Should have species and reactions
        assert model.num_species > 0
        assert model.num_reactions > 0

        # Initial state should be valid
        state = model.initial_state()
        assert len(state) == model.num_species

    def test_example_inverter_file(self):
        """Test loading the example inverter_circuit.sbol file."""
        example_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
            "examples",
            "inverter_circuit.sbol",
        )
        if not os.path.exists(example_path):
            pytest.skip("Example file not found")

        model = parse_sbol_file(example_path)
        assert model.num_species >= 2
        assert model.num_reactions >= 1


class TestSanitizeName:
    """Tests for the name sanitization utility."""

    def test_simple_name(self):
        assert _sanitize_name("GFP") == "GFP"

    def test_name_with_spaces(self):
        assert _sanitize_name("my protein") == "my_protein"

    def test_name_with_special_chars(self):
        assert _sanitize_name("prot-1/A") == "prot_1_A"

    def test_empty_name(self):
        assert _sanitize_name("") == "unnamed"
