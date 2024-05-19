from psychopy import visual, sound, data, event, core, gui
from psychopy import prefs
prefs.hardware['audioLib']= [
                                'sounddevice', 
                                'pyo', 
                                'ptb', 
                                'pygame'
                            ]
import subprocess
import random
import pandas as pd
from datetime import datetime
import csv
import os
import sys
import codecs
import gc
import multiprocessing as mp

class Experiment:
    """
    Parameters
    ----------
    block, block_num: str
        The current block (practice/vv/va/vva) and its order in the sequence of blocks (0/1/2/3)

    Attributes
    ----------
    num_trials: int
        Number of trials in each condition (i.e. - total_trials=num_trials*num_conditions). Needed 
        for initializing PsychoPy's TrialHandler class in the following parameter: 
        nReps - number of repeats for all conditions
    system_id: int
        System identifier specifying which monitor to draw in.
    yes: str
        The keyboard key to respond 'Yes' in the forced choice task and 
        move left on the scale in the Likert task.
    no: str
        The keyboard key to respond 'No' in the forced choice task and 
        move left on the scale in the Likert task.
    accept: str
        The keyboard key to confirm choice in the Likert task.
    win: class
        Initialize PsychoPy Window class on which to display experiment.
    blank_scr: class
        Initialize PsychoPy class to display text for instructions and tasks.
    """
        
    def __init__(self, subject, age, sex, block, block_num):
        self.num_trials = 1
        self.system_id = 0
        self.yes = 'g'
        self.no = 'h'
        self.accept = 'space'
        self.subject = subject
        self.age = age
        self.sex = sex
        self.block = block
        self.block_num = block_num
        self.win = visual.Window(fullscr=True, color='black', units='pix', screen=self.system_id)
        self.text = visual.TextStim(self.win, text=None, alignText="center", color = 'white')


    def create_trials(self):
        trial_list = []
        if self.block == 'practice':
            df = pd.read_csv('./stim/practice_trials.csv')
        else:
            df = pd.read_csv(f'./data/trials/{self.subject}/{self.subject}_{self.block}_trials_modded.csv')
        df = df.sample(frac=1).reset_index()    # shuffle row order
        for index, row in df.iterrows():
            trial_dict = row.to_dict()  # need to provide list of dicts to PsychoPy TrialHandler
            trial_dict['Date'] = datetime.now()
            trial_dict['Subject'] = self.subject
            trial_dict['Age'] = self.age
            trial_dict['Sex'] = self.sex
            trial_dict['TrialNumber'] = index+1
            trial_dict['BlockNumber'] = self.block_num
            trial_dict['Block'] = self.block
            trial_dict['CorrectResp'] = self.yes
            trial_list.append(trial_dict)
        # PsychoPy TrialHandler
        trials = data.TrialHandler(trialList=trial_list, nReps=self.num_trials, method='sequential')
        trials.data.addDataType('Resp')
        trials.data.addDataType('LikertResp')

        return trials
    

    def show_text_and_wait(self, file_name = None, message = None):
        event.clearEvents()
        if message is None:
            with codecs.open (file_name, "r", "utf-8") as file :
                message = file.read()
        self.text.setText(message)
        self.text.draw()
        self.win.flip()
        while True :
            if len(event.getKeys()) > 0:
                core.wait(0.2)
                break
            event.clearEvents()
            core.wait(0.2)
            self.text.draw()
            self.win.flip()


    def make_stim(self, trial):
        mov = visual.MovieStim(
            self.win, 
            trial['VideoPath'],
            size=[1920, 1200],              
            pos=[0, 0],             
            flipVert=False,         
            flipHoriz=False,        
            loop=False,             
            noAudio=True
        )
        mov_aud = sound.Sound(trial['AudioPath'])

        return mov, mov_aud
    

    def forced_choice(self, trials, responded):
        event.clearEvents()
        self.text.setText('A votre avis, est-ce que cette interaction est vraie ? \n \n (vraie: G; fausse: H)')
        self.text.draw()
        self.win.flip()
        keys = event.getKeys([self.yes, self.no, 'x'])
        if self.yes in keys or self.no in keys:
            trials.addData('Resp', keys[0])
            responded = True
        elif 'x' in keys:
            self.win.close()
            core.quit()

        return responded, keys


    def rate_confidence(self, trials, scale, responded):
        resp = ''
        if scale.noResponse:
            self.text.setText('Dans quel point êtes-vous sûr.e.s de votre choix ? \n \n (gauche: G ; droite: H ; confirmer: Espace)')
            self.text.draw()
            scale.draw()
            self.win.flip()
        else:
            resp = scale.getRating()
            trials.addData('LikertResp', resp)
            scale.draw()
            self.win.flip()
            responded = True

        return responded, resp
    

    def write_trial(self, res_file, trial, fc_resp, likert_resp):
        with open(res_file, 'a', newline='') as saveFile:
                fileWriter = csv.writer(saveFile, dialect='excel')
                if os.stat(res_file).st_size == 0: # if file is empty, insert header
                    fileWriter.writerow(('Date', 'Subject', 'Age', 'Sex', 'TrialNumber', 'BlockNumber', 'Block', 'VideoPath', 'AudioPath', 'SpeakerID', 'ListenerID', 'SpeakerDyad', 'ListenerDyad', 'SpeakerExtract', 'ListenerExtract', 'Condition', 'SpeakerSex', 'Duration', 'CorrectResp', 'Resp', 'LikertResp'))
                #write trial
                fileWriter.writerow([trial['Date'], trial['Subject'], trial['Age'], trial['Sex'], trial['TrialNumber'], trial['BlockNumber'], trial['Block'], trial['VideoPath'], trial['AudioPath'], trial['SpeakerID'], trial['ListenerID'], trial['SpeakerDyad'], trial['ListenerDyad'], trial['SpeakerExtract'], trial['ListenerExtract'], 
                                     trial['Condition'], trial['SpeakerSex'], trial['Duration'], trial['CorrectResp'], fc_resp, likert_resp])


    def run(self):
        trials = self.create_trials()
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        res_file = f'./data/responses/{self.subject}/{timestamp}_{self.subject}_responses.csv'

        for i in range(1, 4):
            self.show_text_and_wait(file_name=f'./stim/instructions/instructions_{i}.txt')
        if self.block == 'practice':
            self.show_text_and_wait(file_name='./stim/instructions/practice.txt')
        
        for trial in trials:
            scale = visual.RatingScale(self.win, low=1, high=4, markerStart=2, scale=None, 
                                            labels=['pas du tout sûr', 'complétement certain'], 
                                            leftKeys=[self.yes], rightKeys=[self.no], acceptKeys=[self.accept])

            # Trial
            mov, aud = self.make_stim(trial)
            mov.play()
            if trial['Block'] != 'vv':
                aud.play()         
            while mov.isFinished == False:
                mov.draw()
                self.win.flip()
            
            # Booleans to track if participant has responded or not
            responded_fc = False
            responded_likert = False
            # Do not move forward until participant has responded to the task
            while responded_fc == False:
                responded, keys = self.forced_choice(trials, responded_fc)
                responded_fc = responded       # Switch boolean to True if participant has responded
            while responded_likert == False:
                responded, rating = self.rate_confidence(trials, scale, responded_likert)
                responded_likert = responded

            self.write_trial(res_file, trial, keys[0], rating)


            if trials.getFutureTrial(n=1) != None:
                self.show_text_and_wait(file_name="./stim/instructions/trial_end.txt")
            else:
                if (self.block == 'practice'):
                    self.show_text_and_wait(file_name="./stim/instructions/practice_end.txt")
                elif (self.block_num != 3):
                    self.show_text_and_wait(file_name=f"./stim/instructions/block_{self.block_num}_end.txt")
                elif (self.block_num == 3):
                    self.show_text_and_wait(file_name="./stim/instructions/end.txt")
           
            del scale
            del mov
            del aud
            del trial
            gc.collect()

        self.win.close()
    

if __name__ == '__main__':
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])

    blocks = ['practice'] + random.sample(['va', 'vv', 'vva'], 3)

    subject_info = {u'Subject':'', u'Age':'', u'Sex': u'f/m'}
    dlg = gui.DlgFromDict(subject_info, title=u'SOC-CON')
    if dlg.OK:
        subject = subject_info[u'Subject']
        age = subject_info[u'Age']
        sex = subject_info[u'Sex']
    else:
        core.quit() #the user hit cancel so exit

    for i in range(len(blocks)):
        block = Experiment(subject, age, sex, blocks[i], i)
        block.run()