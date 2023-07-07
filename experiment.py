from psychopy import visual, sound, data, event, core, gui, monitors
from psychopy import prefs
prefs.hardware['audioLib']= [
                                'sounddevice', 
                                'ptb',
                                'pyo', 
                                'pygame'
                            ]
import pandas as pd
from datetime import datetime
import csv
import os
import random
import codecs


class Experiment:
    """
    Parameters
    ----------
    num_trials: int
        Number of trials in each condition (i.e. - total_trials=num_trials*num_conditions). Needed 
        for initializing PsychoPy's TrialHandler class in the following parameter: 
        nReps - number of repeats for all conditions

    Attributes
    ----------
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
    blocks: list
        A list containing the different blocks (VV, VA, VVA) in a random order 
        upon each initialization.
    win: class
        Initialize PsychoPy Window class on which to display experiment.
    blank_scr: class
        Initialize PsychoPy class to display text for instructions and tasks.
    """
        
    def __init__(self, num_trials):
        self.num_trials = num_trials
        self.system_id = 1
        self.yes = 'g'
        self.no = 'h'
        self.accept = 'space'
        self.blocks = random.sample(['vv', 'va', 'vva'], 3)
        self.win = visual.Window(fullscr=True, color='black', screen=self.system_id)
        self.text = visual.TextStim(self.win, text=None, alignText="center", color = 'white')


    def dialog_box(self):
        subject_info = {u'Subject':'', u'Age':'', u'Sex': u'f/m'}
        dlg = gui.DlgFromDict(subject_info, title=u'SOC-CON', screen=self.system_id)
        if dlg.OK:
            subject = subject_info[u'Subject']
            age = subject_info[u'Age']
            sex = subject_info[u'Sex']
        else:
            core.quit() #the user hit cancel so exit

        return subject, age, sex if dlg.OK else None


    def create_practice_trials(self, subject, age, sex):
        pt_list = []
        df_pt = pd.read_csv('./stim/practice_trials.csv')
        for index, row in df_pt.iterrows():
            trial_dict = row.to_dict()  # need to provide list of dicts to PsychoPy TrialHandler
            trial_dict['Date'] = datetime.now()
            trial_dict['Subject'] = subject
            trial_dict['Age'] = age
            trial_dict['Sex'] = sex
            trial_dict['TrialNumber'] = index+1
            trial_dict['CorrectResp'] = self.yes
            pt_list.append(trial_dict)
        # PsychoPy TrialHandler
        practice_trials = data.TrialHandler(trialList=pt_list, nReps=self.num_trials, method='sequential')
        practice_trials.data.addDataType('Resp')
        practice_trials.data.addDataType('LikertResp')

        return practice_trials


    def create_trials(self, subject, age, sex):
        trial_list = []
        for i in range(len(self.blocks)):
            df = pd.read_csv(f'./data/trials/{subject}/{subject}_{self.blocks[i]}_trials_modded.csv')
            df = df.sample(frac=1).reset_index()    # shuffle row order
            for index, row in df.iterrows():
                trial_dict = row.to_dict()  # need to provide list of dicts to PsychoPy TrialHandler
                trial_dict['Date'] = datetime.now()
                trial_dict['Subject'] = subject
                trial_dict['Age'] = age
                trial_dict['Sex'] = sex
                trial_dict['TrialNumber'] = index+1
                trial_dict['BlockNumber'] = i+1
                trial_dict['Block'] = self.blocks[i]
                trial_dict['CorrectResp'] = self.yes if row['Condition'] == 'TRUE' else self.no
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
            # size=[1920, 1200],              
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
            self.text.setText('A quel point êtes-vous sûr.e.s de votre choix ? \n \n (gauche: G ; droite: H ; confirmer: Espace)')
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


    def run_practice(self, practice_trials):
        for pt in practice_trials:
            scale = visual.RatingScale(self.win, low=1, high=4, markerStart=2, scale=None, showAccept=False, 
                                labels=['pas du tout sûr', 'complétement certain'], 
                                leftKeys=[self.yes], rightKeys=[self.no], acceptKeys=[self.accept])

            # Trial
            mov, aud = self.make_stim(pt)
            mov.play()
            aud.play()       
            while mov.status != visual.FINISHED:
                mov.draw()
                self.win.flip()

            # Booleans to track if participant has responded or not
            responded_fc = False
            responded_likert = False
            # Do not move forward until participant has responded to the task
            while responded_fc == False:
                responded, keys = self.forced_choice(practice_trials, responded_fc)
                responded_fc = responded       # Switch boolean to True if participant has responded
            while responded_likert == False:
                responded, rating = self.rate_confidence(practice_trials, scale, responded_likert)
                responded_likert = responded

            self.show_text_and_wait(file_name="./stim/instructions/trial_end.txt")
        self.show_text_and_wait(file_name="./stim/instructions/practice_end.txt")


    def run(self):
        self.win.winHandle.set_visible(False)
        subject, age, sex = self.dialog_box()
        self.win.winHandle.set_visible(True)

        practice_trials = self.create_practice_trials(subject, age, sex)
        trials = self.create_trials(subject, age, sex)
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        res_file = f'./data/responses/{subject}/{timestamp}_{subject}_responses.csv'

        for i in range(1, 4):
            self.show_text_and_wait(file_name=f'./stim/instructions/instructions_{i}.txt')

        self.show_text_and_wait(file_name='./stim/instructions/practice.txt')
        self.run_practice(practice_trials)

        for trial in trials:
            scale = visual.RatingScale(self.win, low=1, high=4, markerStart=2, scale=None, 
                                            labels=['pas du tout sûr', 'complétement certain'], 
                                            leftKeys=[self.yes], rightKeys=[self.no], acceptKeys=[self.accept])

            # Trial
            mov, aud = self.make_stim(trial)
            mov.play()
            if trial['Block'] != 'vv':
                aud.play()         
            while mov.status != visual.FINISHED:
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

            # Give a break if end of block
            if trials.getFutureTrial(n=1) != None:
                if trials.getFutureTrial(n=1)['BlockNumber'] != trial['BlockNumber']:
                    self.show_text_and_wait(file_name=f"./stim/instructions/block_{trial['BlockNumber']}_end.txt")
                else:
                    self.show_text_and_wait(file_name="./stim/instructions/trial_end.txt")
        
        self.show_text_and_wait(file_name="./stim/instructions/end.txt")
        self.win.close()


experiment = Experiment(num_trials = 1)
experiment.run()