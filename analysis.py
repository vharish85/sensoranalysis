sample_rate = 50 # 50hz is the sample rate given by Hackathon team
w1  = pd.read_csv("w1.csv")
w2  = pd.read_csv("w2.csv")


time_w1 = (max(w1.time) - min(w1.time))/1000000
print ("Total time gap for activity %d seconds"%time_w1)

time_w2 = (max(w2.time) - min(w2.time))/1000000
print ("Total time gap for activity %d seconds"%time_w2)

miss_w1 = ((time_w1*50) - w1.shape[0])
print (miss_w1)

miss_w2 = ((time_w2*50) - w2.shape[0])
print (miss_w2)

w1=w1.drop(w1.tail(40).index) # drop last n rows
w2=w2.drop(w2.tail(27).index) # drop last n rows

w1['phi'] = np.degrees(np.arctan2(-1*w1['Ay'],w1['Ax']))
w2['phi'] = np.degrees(np.arctan2(-1*w2['Ay'],w2['Ax']))

w1['secs'] = pd.Series(range(1,1201), index=w1.index)
w2['secs'] = pd.Series(range(1,1201), index=w2.index)

w1['A'] = np.sqrt(w1['Ax']*w1['Ax'] + w1['Ay'] * w1['Ay'] + w1['Az']*w1['Az'])
w1['G'] = np.sqrt(w1['Gx']*w1['Gx'] + w1['Gy'] * w1['Gy'] + w1['Gz']*w1['Gz'])

w2['A'] = np.sqrt(w2['Ax']*w2['Ax'] + w2['Ay'] * w2['Ay'] + w2['Az']*w2['Az'])
w2['G'] = np.sqrt(w2['Gx']*w2['Gx'] + w2['Gy'] * w2['Gy'] + w2['Gz']*w2['Gz'])

w1['pA'] = butter_filter(w1['A'],0.001,Fs,'high')
w1['pA'] = np.abs(w1['pA'])
w1['pA'] = butter_filter(w1['pA'],5,Fs,'low')

w2['pA'] = butter_filter(w2['A'],0.001,Fs,'high')
w2['pA'] = np.abs(w2['pA'])
w2['pA'] = butter_filter(w2['pA'],5,Fs,'low')

print (w2.shape)


color = sns.color_palette()
plt.figure()
w1['sub'] =0
sns_df = pd.melt(w1, id_vars=['secs','sub'], value_vars=['phi'])
sns.tsplot(data=sns_df, time= "secs", unit='sub',value='value', condition='variable')

plt.figure()
w1['sub'] =0
sns_df = pd.melt(w1, id_vars=['secs','sub'], value_vars=['Gx', 'Gy', 'Gz'])
sns.tsplot(data=sns_df, time= "secs", unit='sub',value='value', condition='variable')


plt.show()


slope = pd.Series(np.gradient(w1.A.values), w1.secs, name='slope')
plt.plot(w1.secs,slope,'r')
plt.figure()

slope = pd.Series(np.gradient(w2.A.values), w1.secs, name='slope')
plt.plot(w1.secs,slope)
plt.show()
