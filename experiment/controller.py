from psychopy import core, gui
import random
import sys
import subprocess


blocks = ['practice'] + random.sample(['va', 'vv', 'vva'], 3)

subject_info = {u'Subject':'', u'Age':'', u'Sex': u'f/m'}
dlg = gui.DlgFromDict(subject_info, title=u'SOC-CON')
if dlg.OK:
    subject = subject_info[u'Subject']
    age = subject_info[u'Age']
    sex = subject_info[u'Sex']
else:
    core.quit() #the user hit cancel so exit

subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
subprocess.run(f'python experiment_revised.py {subject} {age} {sex} {blocks[0]} {0}')
subprocess.run(f'python experiment_revised.py {subject} {age} {sex} {blocks[1]} {1}')
subprocess.run(f'python experiment_revised.py {subject} {age} {sex} {blocks[2]} {2}')
subprocess.run(f'python experiment_revised.py {subject} {age} {sex} {blocks[3]} {3}')