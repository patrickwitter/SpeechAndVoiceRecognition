Password = None
USER = None
# import required libraries
import sounddevice as sd
from scipy.io.wavfile import write

def get_audio():  
    # Sampling frequency
    freq = 48000
    
    # Recording duration
    duration = 5
    
    # Start recorder with the given values 
    # of duration and sample frequency
    print("------------------------------Recording started for a period of 5 seconds-------------------------------------------")
    recording = sd.rec(int(duration * freq), 
                    samplerate=freq, channels=1,dtype="int16")
    
    # Record audio for the given number of seconds
    sd.wait()
  
    # This will convert the NumPy array to an audio
    # file with the given sampling frequency
    print("------------------------------------Recording ended------------------------------------------------------------------")
    # write("recording.wav", freq, recording)

    return recording, freq
  


import speech_recognition as spr
import scipy

def recogSpeech(audio, sr):
  if audio is None and sr is None:
    audio ,sr= get_audio()

  # print("Before write audio")
  scipy.io.wavfile.write('recording.wav', sr, audio)
  # Instantiate Recognizer
  recognizer = spr.Recognizer()

  # Convert audio to AudioFile
  # print("Before audio read")
  try:
    clean_support_call = spr.AudioFile('recording.wav')
  except Exception as e: 
    print("An exception occured",e)
  # print("After audio")
  with clean_support_call as source:
      clean_support_call_audio = recognizer.record(source)

  text = recognizer.recognize_google(clean_support_call_audio,
                                    language="en-US")
  print("You said",text)

  return text
import os
import wave
import time
import pickle
import warnings
import numpy as np
from sklearn import preprocessing
from scipy.io.wavfile import read
import python_speech_features as mfcc
from sklearn.mixture import GaussianMixture
def calculate_delta(array):
   
    rows,cols = array.shape
    # print(rows)
    # print(cols)
    deltas = np.zeros((rows,20))
    N = 2
    for i in range(rows):
        index = []
        j = 1
        while j <= N:
            if i-j < 0:
              first =0
            else:
              first = i-j
            if i+j > rows-1:
                second = rows-1
            else:
                second = i+j 
            index.append((second,first))
            j+=1
        deltas[i] = ( array[index[0][0]]-array[index[0][1]] + (2 * (array[index[1][0]]-array[index[1][1]])) ) / 10
    return deltas
def extract_features(audio,rate):
       
    mfcc_feature = mfcc.mfcc(audio,rate, 0.025, 0.01,20,nfft = 1200, appendEnergy = True)    
    mfcc_feature = preprocessing.scale(mfcc_feature)
    # print(mfcc_feature)
    delta = calculate_delta(mfcc_feature)
    combined = np.hstack((mfcc_feature,delta)) 
    return combined
import scipy
# scipy.io.wavfile.write('recording.wav', sr, audio)
def record_audio_train():
  global USER
  Name =(input("Please Enter Your Name:"))
  USER = Name
  TIMES = 20
  audio, sr = get_audio()
  for count in range(TIMES):
    print("Record Voice ",count + 1,"/",TIMES)
    OUTPUT_FILENAME=Name+"-sample"+str(count)+".wav"
    # WAVE_OUTPUT_FILENAME=os.path.join("training_set",OUTPUT_FILENAME)
    # print("WAVEOUTPUT FILE name",WAVE_OUTPUT_FILENAME)
    trainedfilelist = open("training_set_addition.txt", 'a')
    # %cd training_set
    trainedfilelist.write(OUTPUT_FILENAME+"\n")
    scipy.io.wavfile.write(OUTPUT_FILENAME, sr, audio)

import scipy
def record_audio_test(audio,sr):
  if audio is None and sr is None:
    audio , sr = get_audio()
  OUTPUT_FILENAME="sample.wav"
  # WAVE_OUTPUT_FILENAME=os.path.join("testing_set",OUTPUT_FILENAME)
  trainedfilelist = open("testing_set_addition.txt", 'a')
  trainedfilelist.write(OUTPUT_FILENAME+"\n")
  scipy.io.wavfile.write(	OUTPUT_FILENAME, sr, audio)

def train_model():
  #What we want to do is for each speaker to have their own model
  train_file = "training_set_addition.txt"        
  file_paths = open(train_file,'r')
  features = np.asarray(())
  epochs = 100
  speakers = set() # set of speakers
  for path0 in file_paths:
    speakers.add(path0.split("-")[0])
  print("---------------Speakers-----------------",speakers)
  choice = int(input("1. Train for all speakers 2. Train for specific Speaker"))
  if choice == 2:
    speakers = list(speakers)
    # print("speakers new",speakers)
    for i in range(len(speakers)):
      print(i+1,"Name:",speakers[i])
    spC = int(input("Choose the number next to your speaker"))
    x = []
    x.append(speakers[spC-1])
    speakers = x.copy()
    # print("speakers new2",speakers)
    
    
  for speaker in speakers:
    # print("In loop-----------",speaker)
    for e in range(epochs):
      file_paths2 = open(train_file,'r')
      print("Training for Sepeaker--",speaker,"In epoch",e)
      for path in file_paths2: 

          s =  path.split("-")[0]
          # print("Speaker-----------",speaker,"s-----------",s)
          if (s == speaker):
            # print("In second loop")  
            path = path.strip()   
            # print("-----------------------------------------------",path)

            sr,audio = read(path)
          #   print(sr)
            vector   = extract_features(audio,sr)
            
            if features.size == 0:
                features = vector
            else:
                features = np.vstack((features, vector))

    
    gmm = GaussianMixture(n_components = 6, max_iter = 200, covariance_type='diag',n_init = 3)
    gmm.fit(features)
  
  # dumping the trained gaussian model
    picklefile = speaker+".gmm"
    print("-----------------Model File PATH-------------------",picklefile)
    pickle.dump(gmm,open(picklefile,'wb'))
    print('+ modeling completed for speaker:',picklefile," with data point = ",features.shape)   
    features = np.asarray(())
 
def test_model(audio,sr):
  if audio is None and sr is None:
    # print("---------------------Above record test----------------------")
    record_audio_test(None,None)
  else:
    # print("---------------------Audio Given----------------------")
    record_audio_test(audio,sr)

  test_file = "testing_set_addition.txt"      
  file_paths = open(test_file,'r')
    
  gmm_files = [fname for fname in
                os.listdir() if fname.endswith('.gmm')]
  # print("-------------------Length of Files------------------------",len(gmm_files))  
  #Load the Gaussian gender Models
  models    = [pickle.load(open(fname,'rb')) for fname in gmm_files]
  # print("-------------------Length of Models------------------------",len(models)) 
  speakers   = [fname.split("\\")[-1].split(".gmm")[0] for fname 
                in gmm_files]
    
  # Read the test directory and get the list of test audio files 
  for path in file_paths:   
        
      path = path.strip()   
      # print(path)
      sr,audio = read(path)
      vector   = extract_features(audio,sr)
        
      log_likelihood = np.zeros(len(models)) 
      
      for i in range(len(models)):
          gmm    = models[i]  #checking with each model one by one
          scores = np.array(gmm.score(vector))
          log_likelihood[i] = scores.sum()
      
      for i in range(len(models)):
        print("likely-hood of speaker",speakers[i],"--------------",log_likelihood[i])  
      winner = np.argmax(log_likelihood)
      print("\tdetected as - ", speakers[winner])
      return speakers[winner]
    
def setPassword():
  global Password 
  Password = recogSpeech(None,None)
  return 
def checkPassword():
  global Password
  global USER

  USER = "Patrick" if USER == None else USER
  print("Say your password")
  audio , sr = get_audio()
  password = recogSpeech(audio,sr)
  user = test_model(audio,sr)
  print("Actual Password",Password,"Given Password",password,"Detected Speaker",user,"Authorized Speaker",USER)
  if(Password == password and user == USER):
    print("Password is Correct. Welcome ",user,"!")
  elif (Password == password and user != USER):
    print("Password correct. User false, you are ",user)
  elif (Password != password and user == USER and Password != None):
    print("Hey you are correct user",user," but your password is incorrect")
  elif (Password != password and user != USER):
    print("Neither your password is correct and your not the correct user")
  else:
     print("Hey you are correct user",user," but your password is not set")
  
  
#Menu 
while True:
  choice=int(input("\n 1.Record audio for training \n 2.Train Model \n3.Test Model-Voice-Recognition\n4.Test Model-Speech Recognition\n"+
  "5.Set Password\n6.Check Password\nAny other Number-Exit\n"))
  if(choice==1):
    record_audio_train()
  elif(choice==2):
    train_model()
  elif(choice==3):
    test_model(None,None)
  elif(choice==4):
    recogSpeech(None,None)
  elif (choice == 5):
    setPassword()
  elif (choice == 6):
    checkPassword()
  if(choice>6):
    print("Thank you for using our app!")
    break
    