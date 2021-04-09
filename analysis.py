import pandas as pd
import matplotlib

def parse_jb_output(task,noise):
    df = pd.read_csv("summary_"+task+"_rand_"+noise+".tsv",sep="\t",index_col = False,names=["epoch","ds","intent_acc","loss","sem_acc","slot_f1","slot_p","slot_r"])
    df = df.dropna()
    df = df. groupby(["epoch","ds"]).agg({'intent_acc':'mean'})
    df = df.reset_index()
    df_test = df[df['ds'] == 'test']
    df_test = df_test.rename(columns={'intent_acc':"test_acc_rand_"+noise})
    df_train = df[df['ds'] == 'train']
    df_train = df_train.rename(columns={'intent_acc':"train_acc_rand_"+noise})
    return df_test, df_train

task = "atis"
df_test, df_train = parse_jb_output(task,"0")
ax = df_train.plot(x='epoch',y='train_acc_rand_0', figsize=[15,15])
df_test.plot(ax=ax,x='epoch',y='test_acc_rand_0')
for noise in ["10","50"]:
    df_test, df_train = parse_jb_output(task,noise)
    df_test.plot(ax=ax,x='epoch',y='test_acc_rand_'+noise)
    df_train.plot(ax=ax,x='epoch',y='train_acc_rand_'+noise)

task = "snips"
for noise in ["0", "10", "20", "30", "40", "50"]:
    df_test, df_train = parse_jb_output(task, noise)
    df_test = df_test[df_test.epoch == 75]
    df_train = df_train[df_train.epoch == 75]
    print(noise)
    print(df_train)
    print(df_test)
