import os
import glob
import itertools
import pandas as pd
import numpy as np
import librosa
from scipy import stats
from sklearn.preprocessing import StandardScaler
from scipy.signal import resample as sci_resample
from scipy import signal
from scipy.ndimage import gaussian_filter1d
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from mtrf.model import TRF
from mtrf.stats import nested_crossval

class Pipeline:

    sr = 30
    aus = ["AU12","AU14","AU15", "AU17","AU23","AU24","AU25","AU26","AU28","AU43","Pitch"]
    au_gifs = ["https://imotions.com/wp-content/uploads/2022/10/AU12.gif", "https://imotions.com/wp-content/uploads/2022/10/AU14-dimpler.gif", 
                "https://imotions.com/wp-content/uploads/2022/10/AU15.gif", "https://imotions.com/wp-content/uploads/2022/10/AU17.gif", 
                "https://imotions.com/wp-content/uploads/2022/10/AU23-lip-tightener.gif", "https://imotions.com/wp-content/uploads/2022/10/AU24.gif", 
                "https://imotions.com/wp-content/uploads/2022/10/AU25-lips-part.gif", "https://imotions.com/wp-content/uploads/2022/10/AU26-with-25.gif", 
                "https://imotions.com/wp-content/uploads/2022/10/AU28-with-26.gif", "https://imotions.com/wp-content/uploads/2022/10/AU43-eyes-closed.gif",
                "https://imotions.com/wp-content/uploads/2022/10/AU54-head-down.gif"]


    def __init__(self, trf_direction, trf_min_lag, trf_max_lag, regularization, modality, audio_type):
        self.trf_direction = trf_direction
        self.trf_min_lag = trf_min_lag
        self.trf_max_lag = trf_max_lag
        self.regularization = regularization
        self.modality = modality
        self.audio_type = audio_type


    def resample_signal(self, signal, length):
        signal_resampled = sci_resample(signal, len(np.arange(0, length, 1/self.sr)))
        signal_resampled = gaussian_filter1d(signal_resampled, sigma=2)

        standardize = StandardScaler()
        signal_resampled = standardize.fit_transform(signal_resampled.reshape(-1, 1))

        return signal_resampled.flatten()
    

    def get_rms(self, file, length):
        audio, audio_sr = librosa.load(file)
        rms_win = 0.5
        rms_hop = 1/self.sr 
        rms = librosa.feature.rms(y=audio, frame_length=int(self.sr*rms_win), hop_length=int(self.sr*rms_hop))
        rms=rms[0]
        rms_resampled = self.resample_signal(rms, length)

        return rms_resampled
    
    
    def get_an_response(self, file, length):  
        file = file.replace(os.sep, '/')
        df = pd.read_csv(file, usecols=[1])
        an = df.to_numpy().flatten()
        an_resampled = self.resample_signal(an, length)

        return an_resampled
    

    def get_au(self, file, au, length, *args):
        if self.modality == 'va':
            au_data = pd.read_csv(file)[au].to_numpy()
        else:
            au_data = pd.read_csv(file)[au].to_numpy()[args[0]::2]

        au_data_resampled = self.resample_signal(au_data, length)

        return au_data_resampled


    def get_trial_info(self, df, idx):
        speaker_id = df.iloc[idx]['SpeakerID']
        listener_id = df.iloc[idx]['ListenerID']
        video_path = df.iloc[idx]['VideoPath']
        audio_path = df.iloc[idx]['AudioPath']
        disp_dyad = df.iloc[idx]['DisplayedDyad']
        duration = df.iloc[idx]['Duration']
        cond = df.iloc[idx]['VideoPath'].split('/')[3]
        au_path = f"./data/aus_pure/{self.modality}/{cond[:-3] if cond == 'fake_ad' else cond}/{disp_dyad}_{video_path.split(os.sep)[-1][:-4]}_aus.csv"
        an_path = audio_path[:-10]+'.csv'
        an_path = an_path.replace('audio', 'audio_carney')

        trial_dict = {'video_path': video_path, 'audio_path': audio_path, 'an_path': an_path, 'au_path': au_path, 
                      'displayed_dyad': disp_dyad, 'duration': duration, 'condition': cond, 'speaker_id': speaker_id, 'listener_id': listener_id}
        
        return trial_dict
    

    def format_data(_self, df, *args):
        all_stim, all_resp, all_conditions, all_trials, all_durations, all_audios = [], [], [], [], [], []

        for i in range(len(df)):
            trial = _self.get_trial_info(df, i)
            if os.path.exists(trial.get('au_path')):
                if _self.modality == 'va':
                    if _self.audio_type == 'auditory_nerve':
                        stim = _self.get_an_response(trial.get('an_path'), trial.get('duration'))
                    else:
                        stim = _self.get_rms(trial.get('audio_path'), trial.get('duration'))
                    resp = _self.get_au(trial.get('au_path'), args[0], trial.get('duration'))
                elif _self.modality == 'vv':
                    stim = _self.get_au(trial.get('au_path'), args[0], trial.get('duration'), 0)
                    resp = _self.get_au(trial.get('au_path'), args[1], trial.get('duration'), 1)

            all_stim.append(stim)
            all_resp.append(resp)
            all_conditions.append('true' if trial.get('condition') == "true" else 'fake')
            all_trials.append(trial.get('video_path'))
            all_durations.append(trial.get('duration'))
            all_audios.append(trial.get('audio_path') if _self.audio_type=='rms' else trial.get('an_path'))

        return all_stim, all_resp, all_conditions, all_trials, all_durations, all_audios
    

    def train_model(_self, df):
        au_trfs = []
        regularization = np.logspace(-1, 6, 10)
        for item in itertools.product(_self.aus, repeat=1 if _self.modality=='va' else 2):
            stim, resp, conds, trials, durations, audios = _self.format_data(df, *item)
            trf = TRF(direction=_self.trf_direction)
            r_unbiased, best_regularization = nested_crossval(trf, stim, resp, _self.sr, _self.trf_min_lag, _self.trf_max_lag, regularization, k=5, verbose=False)
            # trf.train(stimulus=stim, response=resp, fs=_self.sr, tmin=_self.trf_min_lag, tmax=_self.trf_max_lag, regularization=_self.regularization)
            trf.train(stimulus=stim, response=resp, fs=_self.sr, tmin=_self.trf_min_lag, tmax=_self.trf_max_lag, regularization=best_regularization)
            au_trfs.append(trf)

        return au_trfs
    

    def predict_response(_self, df, trfs):
        data =  {
                    'listener_au': [],
                    'condition': [], 
                    'trial': [],
                    'duration': [],
                    'audio': [],
                    'r': [],
                    'r2': [], 
                    'mae': [], 
                    'mse': [], 
                    'rmse': [], 
                    'actual': [], 
                    'predicted': []
                }
        if _self.modality == 'vv':
            data['speaker_au'] = []

        for idx, item in enumerate(itertools.product(_self.aus, repeat=1 if _self.modality=='va' else 2)):
            stim, resp, conds, trials, durations, audios = _self.format_data(df, *item)
            for i, row in enumerate(zip(stim, resp, conds, trials, durations, audios)):
                response = row[1]
                prediction, correlation = trfs[idx].predict(stimulus=row[0], response=response)
                prediction = np.asarray(prediction).flatten()

                r = stats.pearsonr(response, prediction).correlation
                r2 = r2_score(response, prediction)
                mae = mean_absolute_error(response, prediction)
                mse = mean_squared_error(response, prediction, squared=True)
                rmse = mean_squared_error(response, prediction, squared=False)

                data['listener_au'].append(item[0])
                if _self.modality == 'vv':
                    data['speaker_au'].append(item[1])
                data['condition'].append(row[2])
                data['trial'].append(row[3])
                data['duration'].append(row[4])
                data['audio'].append(row[5])
                data['r'].append(r)
                data['r2'].append(r2)
                data['mae'].append(mae)
                data['mse'].append(mse)
                data['rmse'].append(rmse)
                data['actual'].append(response)
                data['predicted'].append(prediction)

        return data
    

    def cross_correlation(self, trial, *args):
        if self.modality == 'va':
            if self.audio_type == 'auditory_nerve':
                stim = self.get_an_response(trial['an_path'], trial['duration'])
            else:
                stim = self.get_rms(trial['audio_path'], trial['duration'])
            resp = self.get_au(trial['au_path'], args[0], trial['duration'])
        else:
            stim = self.get_au(trial['au_path'], args[0], trial['duration'], 0)
            resp = self.get_au(trial['au_path'], args[1], trial['duration'], 1)

        correlation = signal.correlate(stim, resp, mode="full")
        lags = signal.correlation_lags(stim.size, resp.size, mode="full")
        lag = lags[np.argmax(correlation)]
        peak = np.max(correlation)

        return correlation, lags, lag, peak
    

    def make_crosscorr_df(_self, df):
        data = {
                    'displayed_dyad': [],
                    'condition': [],
                    'listener_au': [],
                    'peak_lag': [],
                    'peak': []
                }
        if _self.modality == 'vv':
            data['speaker_au'] = []

        for idx, item in enumerate(itertools.product(_self.aus, repeat=1 if _self.modality=='va' else 2)):
            for i in range(len(df)):
                trial = _self.get_trial_info(df, i)
                if os.path.exists(trial.get('au_path')):
                    corr, lags, peak_lag, peak = _self.cross_correlation(trial, *item)
                    data['displayed_dyad'].append(trial['displayed_dyad'])
                    data['condition'].append('true' if trial.get('condition') == 'true' else 'fake')
                    if _self.modality == 'vv':
                        data['speaker_au'].append(item[0])
                        data['listener_au'].append(item[1])
                    else:
                        data['listener_au'].append(item[0])
                    data['peak_lag'].append(peak_lag/_self.sr)
                    data['peak'].append(peak)

        df_crossCorr = pd.DataFrame(data=data)

        return df_crossCorr
            

    def make_main_df(self):
        df = pd.read_csv('./stim/all_trials_dispDyad.csv')
        df = df[(df['Modality'] == f"{self.modality}")].reset_index(drop=True).drop('Unnamed: 0', axis='columns')

        return df
    

    def make_trf_df(self, data):
        df_trf = pd.DataFrame(data=data)

        df_trf['r'] = df_trf['r'].astype('float')
        df_trf['r2'] = df_trf['r2'].astype('float')
        df_trf['mae'] = df_trf['mae'].astype('float')
        df_trf['mse'] = df_trf['mse'].astype('float')
        df_trf['rmse'] = df_trf['rmse'].astype('float')
        df_trf['condition_detailed'] = df_trf['trial'].str.split('/').str[3]
        df_trf['displayed_dyad'] = df_trf['trial'].str.split('/').str[-1].str.split('\\').str[0]

        return df_trf
    

    def make_response_df(self):
        lst_df = []
        for dir in (glob.glob(f'./data/responses/*')):
            if len(os.listdir(dir)) != 0:
                files = glob.glob(dir+'/*.csv')
                for i in range(len(files)):
                    if len(files) == 4:
                        if i != 0:
                            temp_df = pd.read_csv(files[i])
                            lst_df.append(temp_df)
                    else:
                        temp_df = pd.read_csv(files[i])
                        lst_df.append(temp_df)
        df = pd.concat(lst_df)
        df = df[df['Block'] == f"{self.modality}"]
        df = df.reset_index()
        df['CorrectResp'] = df['Condition'].apply(lambda x: 'g' if x=='TRUE' else 'h')
        df['Accuracy'] = (df['CorrectResp']==df['Resp'])
        df['DisplayedDyad'] = df['AudioPath'].str.split('/').str[4]
        df[['Age','Sex']] = df[['Sex','Age']].where(df['Subject'] > 5, df[['Age','Sex']].values)

        return df


    def get_stats(self, df, mode):
        hits = len(df[(df['Condition'] == 'TRUE') & (df['Resp'] == 'g')])
        correct_rejections = len(df[(df['Condition'] != 'TRUE') & (df['Resp'] == 'h')])

        if mode == 'sdt':
            misses = len(df[(df['Condition'] == 'TRUE') & (df['Resp'] == 'h')])
            false_alarms = len(df[((df['Condition'] != 'TRUE') & (df['Resp'] == 'g'))])

            hit_rate = hits / (hits+misses)
            fa_rate = false_alarms / (false_alarms+correct_rejections)

            z_hitRate = stats.norm.ppf(hit_rate)
            z_falseAlarmRate = stats.norm.ppf(fa_rate)
            d_prime = z_hitRate - z_falseAlarmRate

            num_target_stim = df['Condition'].value_counts()['TRUE']                        # number of TRUE trials
            num_uses_response = df['Resp'].value_counts()['g']                              # number of times participants responded TRUE
            unbiased_hit_rate = (hits/num_target_stim) * (hits)/(num_uses_response)

            res = pd.Series({'Hit Rate':hit_rate, 'False Alarm Rate':fa_rate, 'd prime':d_prime, 'Unbiased Hit Rate':unbiased_hit_rate})
        elif mode == 'accuracy':
            res = (hits+correct_rejections)/len(df)

        return res
    

    def ttests(self, df, similarity_measure):
        t_statistics = []
        p_vals = []
        for au in self.aus:
            true_correlations = df[(df['listener_au']==au) & (df['condition']=='true')][similarity_measure].to_list()
            fake_correlations = df[(df['listener_au']==au) & (df['condition']=='fake')][similarity_measure].to_list()
            res = stats.ttest_ind(true_correlations, fake_correlations)
            t_statistics.append(res.statistic)
            p_vals.append(res.pvalue)

        return t_statistics, p_vals
    

    def tolerant_mean(self, arrs):
        lens = [len(i) for i in arrs]
        arr = np.ma.empty((np.max(lens), len(arrs)))
        arr.mask = True
        for idx, l in enumerate(arrs):
            arr[:len(l), idx] = l

        return arr.mean(axis = -1), arr.std(axis=-1)