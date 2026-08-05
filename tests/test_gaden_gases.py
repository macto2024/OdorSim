"""Integration checks for OdorSim's tracked gaden_core gas extension."""

from __future__ import annotations

import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CORE = ROOT / "gaden" / "gaden_common" / "third_party" / "gaden_core"
HEADER = CORE / "include" / "gaden" / "datatypes" / "GasTypes.hpp"
PATCH = ROOT / "patches" / "gaden_core_add_excel_gases.patch"


class GadenGasIntegrationTests(unittest.TestCase):
    def test_tracked_patch_defines_extended_gases(self):
        text = PATCH.read_text(encoding="utf-8")
        for gas_id, gas in enumerate(
            ("methanol", "acetic_acid", "ethyl_acetate", "toluene"),
            start=14,
        ):
            self.assertIn(f"{gas}", text)
            self.assertIn(f"= {gas_id}", text)

    @unittest.skipUnless(HEADER.is_file(), "local gaden_core clone is not installed")
    def test_tracked_patch_matches_local_gaden_core(self):
        self.assertTrue(PATCH.is_file())
        completed = subprocess.run(
            ["git", "-C", str(CORE), "apply", "--reverse", "--check", str(PATCH)],
            capture_output=True,
            text=True,
        )
        self.assertEqual(
            completed.returncode,
            0,
            completed.stderr or "tracked gas patch does not match local gaden_core",
        )

    @unittest.skipUnless(shutil.which("g++"), "g++ is required for the C++ smoke test")
    @unittest.skipUnless(HEADER.is_file(), "local gaden_core clone is not installed")
    def test_cpp_enum_names_and_specific_gravity(self):
        source = r"""
#include <array>
#include <iostream>
#include <gaden/datatypes/GasTypes.hpp>

int main()
{
    constexpr std::array<gaden::GasType, 4> types = {
        gaden::GasType::methanol,
        gaden::GasType::acetic_acid,
        gaden::GasType::ethyl_acetate,
        gaden::GasType::toluene,
    };
    static_assert(static_cast<int>(types[0]) == 14);
    static_assert(static_cast<int>(types[1]) == 15);
    static_assert(static_cast<int>(types[2]) == 16);
    static_assert(static_cast<int>(types[3]) == 17);
    static_assert(gaden::SpecificGravity.size() == 18);
    for (const auto type : types)
    {
        const auto index = static_cast<std::size_t>(type);
        if (gaden::to_string(type).empty() || gaden::SpecificGravity.at(index) <= 0)
            return 1;
        std::cout << gaden::to_string(type) << " "
                  << gaden::SpecificGravity.at(index) << "\n";
    }
    return 0;
}
"""
        with tempfile.TemporaryDirectory(prefix="odorsim_gas_cpp_") as temp:
            temp_path = Path(temp)
            cpp = temp_path / "gas_smoke.cpp"
            binary = temp_path / "gas_smoke"
            cpp.write_text(source, encoding="ascii")
            subprocess.run(
                ["g++", "-std=c++17", "-I", str(CORE / "include"), str(cpp), "-lfmt", "-o", str(binary)],
                check=True,
            )
            completed = subprocess.run([str(binary)], check=True, capture_output=True, text=True)
        self.assertEqual(
            [line.split()[0] for line in completed.stdout.splitlines()],
            ["methanol", "acetic_acid", "ethyl_acetate", "toluene"],
        )
