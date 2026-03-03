"""Unit tests for GenBank import module."""

import os
import tempfile

import pytest

from bioprover.models.genbank_import import (
    GenBankImporter,
    parse_genbank_file,
    parse_genbank_string,
    _sanitize_name,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

TOGGLE_SWITCH_GB = """\
LOCUS       toggle_switch       4500 bp    DNA     circular SYN
DEFINITION  Genetic toggle switch.
FEATURES             Location/Qualifiers
     promoter        1..100
                     /gene="pTet"
                     /note="TetR-repressible promoter"
     CDS             131..1200
                     /gene="LacI"
                     /product="LacI repressor protein"
     terminator      1201..1300
                     /gene="T1"
     regulatory      1301..1400
                     /regulatory_class="repressor_binding_site"
                     /gene="TetR"
                     /note="TetR represses LacI"
                     /bound_moiety="TetR"
     promoter        1501..1600
                     /gene="pLac"
                     /note="LacI-repressible promoter"
     CDS             1631..2500
                     /gene="TetR"
                     /product="TetR repressor protein"
     terminator      2501..2600
                     /gene="T2"
     regulatory      2601..2700
                     /regulatory_class="repressor_binding_site"
                     /gene="LacI"
                     /note="LacI represses TetR"
                     /bound_moiety="LacI"
ORIGIN
        1 atgcatgcat gcatgcat
//
"""

MINIMAL_GB = """\
LOCUS       minimal       1000 bp    DNA     linear SYN
DEFINITION  Minimal GenBank with one CDS.
FEATURES             Location/Qualifiers
     CDS             1..900
                     /gene="GFP"
                     /product="green fluorescent protein"
ORIGIN
        1 atgcatgcat
//
"""

NO_FEATURES_GB = """\
LOCUS       empty       500 bp    DNA     linear SYN
DEFINITION  No features at all.
ORIGIN
        1 atgcatgcat
//
"""


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestGenBankImporter:
    """Tests for the GenBankImporter class."""

    def test_parse_toggle_switch(self):
        """Import a toggle switch from GenBank string."""
        importer = GenBankImporter()
        model = importer.import_string(TOGGLE_SWITCH_GB, name="toggle")

        assert model.name == "toggle"
        assert model.num_species >= 2  # LacI_protein and TetR_protein
        assert model.num_reactions >= 2  # at least production reactions

    def test_toggle_has_repression(self):
        """The toggle switch should have Hill repression reactions."""
        from bioprover.models.reactions import HillRepression

        model = parse_genbank_string(TOGGLE_SWITCH_GB, name="toggle")

        hill_reactions = [
            r for r in model.reactions
            if isinstance(r.kinetic_law, HillRepression)
        ]
        assert len(hill_reactions) >= 1

    def test_toggle_has_degradation(self):
        """All protein species should have degradation reactions."""
        from bioprover.models.reactions import LinearDegradation

        model = parse_genbank_string(TOGGLE_SWITCH_GB, name="toggle")

        deg_reactions = [
            r for r in model.reactions
            if isinstance(r.kinetic_law, LinearDegradation)
        ]
        assert len(deg_reactions) >= 2

    def test_parse_minimal(self):
        """A minimal GenBank with one CDS produces a model with one protein."""
        model = parse_genbank_string(MINIMAL_GB, name="minimal")

        assert model.num_species >= 1
        species_names = model.species_names
        assert any("GFP" in n for n in species_names)

    def test_minimal_constitutive(self):
        """A CDS without regulation gets constitutive production."""
        from bioprover.models.reactions import ConstitutiveProduction

        model = parse_genbank_string(MINIMAL_GB, name="minimal")

        const_reactions = [
            r for r in model.reactions
            if isinstance(r.kinetic_law, ConstitutiveProduction)
        ]
        assert len(const_reactions) >= 1

    def test_no_features_warns(self):
        """Missing FEATURES section emits a warning."""
        importer = GenBankImporter()
        model = importer.import_string(NO_FEATURES_GB, name="empty")

        assert model.num_species == 0
        assert len(importer.warnings) >= 1

    def test_import_file(self, tmp_path):
        """Test importing from a file."""
        gb_file = tmp_path / "test.gb"
        gb_file.write_text(TOGGLE_SWITCH_GB, encoding="utf-8")

        model = parse_genbank_file(str(gb_file))
        assert model.num_species >= 2

    def test_import_file_not_found(self):
        """FileNotFoundError for missing file."""
        with pytest.raises(FileNotFoundError):
            parse_genbank_file("/nonexistent/path/model.gb")

    def test_model_is_simulatable(self):
        """The generated model should be simulatable."""
        model = parse_genbank_string(TOGGLE_SWITCH_GB, name="toggle")

        assert model.num_species > 0
        assert model.num_reactions > 0

        state = model.initial_state()
        assert len(state) == model.num_species

    def test_example_toggle_switch_file(self):
        """Test loading the example toggle_switch.gb file."""
        example_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
            "examples",
            "toggle_switch.gb",
        )
        if not os.path.exists(example_path):
            pytest.skip("Example file not found")

        model = parse_genbank_file(example_path)
        assert model.num_species >= 2
        assert model.num_reactions >= 2


class TestGenBankSanitizeName:
    """Tests for the name sanitization utility."""

    def test_simple_name(self):
        assert _sanitize_name("GFP") == "GFP"

    def test_name_with_spaces(self):
        assert _sanitize_name("my protein") == "my_protein"

    def test_name_with_special_chars(self):
        assert _sanitize_name("prot-1/A") == "prot_1_A"

    def test_empty_name(self):
        assert _sanitize_name("") == "unnamed"
