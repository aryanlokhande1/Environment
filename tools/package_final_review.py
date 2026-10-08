"""Package the validated medium candidate and preserved superseded full evidence."""
import argparse,hashlib,io,json,subprocess,tarfile
from pathlib import Path
from environment.persistence.checkpoint import file_hash

parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
root=Path.cwd();A=Path('data/output/finalization-validation');gate=json.loads((A/'medium-gap-outcome-gate.json').read_text())
assert gate['status']=='PASS' and gate['bundle_sha256']==file_hash(Path('artifacts/gold_events_v3/manifest.json'))
if args.output.exists():raise ValueError('do not overwrite a review package')
files=set()
def include(path):
    p=Path(path)
    if p.is_dir():
        for f in p.rglob('*'):
            if f.is_file() and not f.is_symlink() and '__pycache__' not in f.parts and f.suffix!='.pyc':files.add(f)
    elif p.is_file():files.add(p)
for p in ['README.md','pyproject.toml','.gitattributes','.gitignore','src','tests','tools','docs','config','artifacts/gold_events_v3',A/'full-run-bundle-before-gap-fix',A/'full/may-corrected-v2-final-20260502',A/'medium'/gate['run_id']]:include(p)
for extension in ['*.json','*.log','*.md']:
    for p in A.glob(extension):include(p)
for folder in ['medium-gap-outcome-metrics','full-metrics','historical-v3-competing','historical-v3-subpop']:
    for p in (A/folder).glob('*.json'):include(p)
for p in [A/'medium-gap-outcome-metrics/simulated_pl.parquet',A/'full-metrics/simulated_pl.parquet',Path('data/output/corrected-v1-validation/phase2-metrics.json')]:include(p)
inventory={'status':'MEDIUM_VALIDATED_FULL_CANDIDATE_PENDING_APPROVAL','base_commit':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
    'candidate_gate':gate,'files_sha256':{str(p):file_hash(p) for p in sorted(files)},
    'superseded_full_reference':{'run_id':'may-corrected-v2-final-20260502','bundle':'data/output/finalization-validation/full-run-bundle-before-gap-fix','certifies_current_candidate':False},
    'limitations':['Current candidate still needs full-May validation; original ONE-run cap requires approval for a second full run.','Run manifests retain original local absolute input/bundle paths; relocate operational paths explicitly when restoring elsewhere.','Immutable historical source files are referenced by SHA, not duplicated in this archive.']}
manifest=json.dumps(inventory,indent=2,sort_keys=True).encode()+b'\n'
readme=b'''Current corrected-v2 candidate: medium fidelity/correctness/contract validated.
Full validation of this exact identity bundle is PENDING APPROVAL under the
user's original ONE final full-May run limit. The completed full run and its
exact pre-gap bundle are preserved as SUPERSEDED reference evidence only.
Read data/output/finalization-validation/final-report.md first.
PACKAGE_MANIFEST.json identifies every included file and SHA256.
No raw immutable historical sources, secrets, .git directory or virtualenv
are included. Restore against the recorded base commit. Run manifests retain
original operational absolute paths; adjust those explicitly when relocating.
No commit, push, S3/PostgreSQL write or Transformer/RL integration was performed.
'''
args.output.parent.mkdir(parents=True,exist_ok=True)
with tarfile.open(args.output,'w:gz',compresslevel=1) as archive:
    for p in sorted(files):archive.add(p,arcname=str(p),recursive=False)
    for name,data in [('PACKAGE_MANIFEST.json',manifest),('PACKAGE_README.txt',readme)]:
        info=tarfile.TarInfo(name);info.size=len(data);archive.addfile(info,io.BytesIO(data))
with tarfile.open(args.output,'r:gz') as archive:
    for name,sha in inventory['files_sha256'].items():
        stream=archive.extractfile(name);digest=hashlib.sha256()
        for block in iter(lambda:stream.read(1024*1024),b''):digest.update(block)
        assert digest.hexdigest()==sha,name
proof={'status':'PASS','package':str(args.output),'sha256':file_hash(args.output),'bytes':args.output.stat().st_size,'files_verified':len(files),'candidate_bundle_sha256':gate['bundle_sha256'],'readiness':inventory['status']}
args.output.with_suffix(args.output.suffix+'.validation.json').write_text(json.dumps(proof,indent=2)+'\n');print(json.dumps(proof,indent=2))
