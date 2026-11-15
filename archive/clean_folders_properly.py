import os
import glob

PIPELINE_ROOT = r"."

# 1. Clean folder 1
f1_outputs = os.path.join(PIPELINE_ROOT, "1", "outputs")
if os.path.exists(f1_outputs):
    for pat in ["duct_*", "pipe_*", "pipe_turb_*", "turb_*"]:
        for f in glob.glob(os.path.join(f1_outputs, pat)):
            os.remove(f)
            print("Removed from 1/outputs:", os.path.basename(f))

f1_figures = os.path.join(PIPELINE_ROOT, "1", "figures")
if os.path.exists(f1_figures):
    for f in glob.glob(os.path.join(f1_figures, "*")):
        os.remove(f)
        print("Removed from 1/figures:", os.path.basename(f))

# 2. Clean folder 2
f2_outputs = os.path.join(PIPELINE_ROOT, "2", "outputs")
if os.path.exists(f2_outputs):
    for pat in ["duct_*", "turb_*", "pipe_turb_*", "flowfield_pinn_*", "flowfield_comparison_streamlines.png", "error_domain_u_linear.png", "physics_pde_residuals_map.png", "entire_domain_error_vs_Re_linear.png"]:
        for f in glob.glob(os.path.join(f2_outputs, pat)):
            os.remove(f)
            print("Removed from 2/outputs:", os.path.basename(f))

f2_figures = os.path.join(PIPELINE_ROOT, "2", "figures")
if os.path.exists(f2_figures):
    for f in glob.glob(os.path.join(f2_figures, "*")):
        os.remove(f)
        print("Removed from 2/figures:", os.path.basename(f))

# 3. Clean folder 3
f3_figures = os.path.join(PIPELINE_ROOT, "3", "figures")
if os.path.exists(f3_figures):
    for pat in ["pipe_*", "pipe_turb_*", "turb_*", "flowfield_pinn_*"]:
        for f in glob.glob(os.path.join(f3_figures, pat)):
            os.remove(f)
            print("Removed from 3/figures:", os.path.basename(f))

f3_outputs = os.path.join(PIPELINE_ROOT, "3", "outputs")
if os.path.exists(f3_outputs):
    for pat in ["pipe_*", "pipe_turb_*", "turb_*", "flowfield_pinn_*"]:
        for f in glob.glob(os.path.join(f3_outputs, pat)):
            os.remove(f)
            print("Removed from 3/outputs:", os.path.basename(f))

# 4. Clean folder 4
f4_outputs = os.path.join(PIPELINE_ROOT, "4", "outputs")
if os.path.exists(f4_outputs):
    for pat in ["duct_*", "pipe_*", "pipe_turb_*", "flowfield_pinn_*"]:
        for f in glob.glob(os.path.join(f4_outputs, pat)):
            os.remove(f)
            print("Removed from 4/outputs:", os.path.basename(f))

# 5. Clean folder 5
f5_outputs = os.path.join(PIPELINE_ROOT, "5", "outputs")
if os.path.exists(f5_outputs):
    for pat in ["duct_*", "turb_*", "flowfield_pinn_*", "pipe_flowfield_*"]:
        for f in glob.glob(os.path.join(f5_outputs, pat)):
            os.remove(f)
            print("Removed from 5/outputs:", os.path.basename(f))

f5_figures = os.path.join(PIPELINE_ROOT, "5", "figures")
if os.path.exists(f5_figures):
    for f in glob.glob(os.path.join(f5_figures, "*")):
        os.remove(f)
        print("Removed from 5/figures:", os.path.basename(f))

# Clean temporary scratch scripts in folder 5
scratch_scripts = [
    "errror_maps.py", "selftest.py", "fix_all_plots.py", "fix_resub.py",
    "patch_mega_report.py", "apply_reference_updates.py", "assemble_final_paper.py",
    "assemble_stage_by_stage_mega_paper.py", "build_reorganized_mega_paper.py",
    "build_stage_by_stage_mega_paper.py", "generate_mega_report.py",
    "generate_pedagogical_report.py", "make_stage1_to_3.py", "make_stage_sections.py",
    "stage1_block.py", "stage2_block.py", "stage3_block.py", "stage4_block.py",
    "stage5_block.py", "synthesis_block.py", "verify_full_mega_report.py",
    "apply_perfect_revisions.py", "audit_pdf_pages.py", "audit_pages_fast.py"
]
for scr in scratch_scripts:
    p = os.path.join(PIPELINE_ROOT, "5", scr)
    if os.path.exists(p):
        os.remove(p)
        print("Removed scratch script from 5:", scr)

print("\nALL FOLDERS CLEANED AND PURGED OF WRONG/DUPLICATE/SCRATCH FILES!")

