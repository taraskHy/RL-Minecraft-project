import runpy, sys
from pathlib import Path
ns = runpy.run_path(str(Path(__file__).with_name('light_update_jarscan.py')), run_name='lib')  # reuses the parser and prints the first scan again
classes, instructions = ns['classes'], ns['instructions']
print('\n==== implementations of aqg$b.a(Ldfp;Z)Ljava/util/List; (PlayerProvider.getPlayers(pos, boundaryOnly))')
for cname, cf in classes.items():
    for m in cf.methods:
        if m['name'] == 'a' and m['desc'] == '(Ldfp;Z)Ljava/util/List;' and m['code']:
            print('--', cname)
            for pc, op, arg in instructions(cf, m['code']):
                print(f'   {pc:5d} {op:16s} {arg if arg is not None else ""}')
