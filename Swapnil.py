import streamlit as st
import joblib
import sklearn
# In[88]:


import matplotlib.pyplot as plt
import pandas as pd
import time
import seaborn as sns
from textblob import TextBlob
import emoji
def get_():
    gv = open("gv.pkl","rb")
    gvk = joblib.load(gv)
    g= open("now.pkl","rb")
    gender= joblib.load(g)
    return (gvk,gender)
def main():
    st.markdown("<h1 style='text-align: center;'>Hello World</h1>", unsafe_allow_html=True)
    st.subheader("Choose option")
    activities = ['Select','Sentimental Analysis','Gender prediction']
    choice = st.selectbox("Select",activities)
    if choice=='Sentimental Analysis':
        st.header("Sentimental analysis")
        st.info("**NOTE**:  \n\n1: Avoid spell mistakes for better result  \n\n2: Use **English** language only")
        x=st.text_input("Enter your statement here")
        blob = TextBlob(x)
        result = blob.sentiment.polarity
        if st.button("Predict"):
            with st.spinner('Wait for it...'):
                time.sleep(5)
    
            P=[]
            if result<0:
                st.write(emoji.emojize(':disappointed:',use_aliases=True))
                st.markdown("**Your expression polarity tells us that you are in bad mood**")
                if (-result)<0.5:
                    for _ in range(int(-result*100)):
                        P.append("POSITIVE")
                    for _ in range(100-int(-result*100)):
                        P.append("NEGATIVE")
                elif (-result)>0.5:
                    for _ in range(int(-result*100)):
                        P.append("NEGATIVE")
                    for _ in range(100-int(-result*100)):
                        P.append("POSITIVE")
                else:
            
                    for _ in range(100-int(-result*100)):
                        P.append("NEGATIVE")
                df=pd.DataFrame({'Copyright: 2020 Swapnil Adnak':P})
                sns.countplot(df['Copyright: 2020 Swapnil Adnak'])
                plt.ylabel('Intensivity')
                st.pyplot()
            elif result>0:
                st.write(emoji.emojize(':smile:',use_aliases=True))
                st.markdown("**Woww Your expression polarity tells us that you are in good mood**")
                if (result)<0.5:
                    for _ in range(int(result*100)):
                        P.append("NEGATIVE")
                    for _ in range(100-int(result*100)):
                        P.append("POSITIVE")
                elif (result)>0.5:
                    for _ in range(int(result*100)):
                        P.append("POSITIVE")
                    for _ in range(100-int(result*100)):
                        P.append("NEGATIVE")
                else:
                    for _ in range(int(result*100)):
                        P.append("POSITIVE")
            

                df=pd.DataFrame({'Copyright: 2020 Swapnil Adnak':P})
                sns.countplot(df['Copyright: 2020 Swapnil Adnak'])
                plt.ylabel('Intensivity')
                st.pyplot()
            else:
                st.write(emoji.emojize(':expressionless:',use_aliases=True))
                st.markdown("**Your expression polarity tells us that your are in neutral mood,nor good nor bad**")
    if choice=='Gender prediction':
        k=get_()
        st.header("Gender prediction")
        x=st.text_input("Enter First name here")
        if st.button("Predict"):
            with st.spinner('Wait for it... '):
                time.sleep(3)
            kk=k[0].transform([x]).toarray()
            result=k[1].predict(kk)
            if (result[0]=='f') | (x=='Vividha') | (x=='vividha'):
                st.markdown("<h3 style='text-align: center;'>Hello Sister</h3>", unsafe_allow_html=True)
                st.image('female.jpg', width=None)
                st.markdown("<h4 style='text-align: center;'>Probability prediction chart of model</h4>", unsafe_allow_html=True)
                chart_data = pd.DataFrame(k[1].predict_proba(kk)*100,columns=["Female",'Male'],index=['Female'])
                st.bar_chart(chart_data,width=5)
            elif result[0]=='m':
                                
                st.markdown("<h3 style='text-align: center;'>Hello brother</h3>", unsafe_allow_html=True)
                st.image('male.jpg', width=None)
                st.markdown("<h4 style='text-align: center;'>Probability prediction chart of model</h4>", unsafe_allow_html=True)
                chart_data = pd.DataFrame(k[1].predict_proba(kk)*100,columns=["Female",'Male'],index=['Male'])
                st.bar_chart(chart_data,width=5)
                
		
	


    st.markdown("<br><br>",unsafe_allow_html=True)
    st.info(" \n\n\nCopyright "+emoji.emojize(':copyright:',use_aliases=True)+" : 2020 Swapnil Adnak")

if __name__ == '__main__':
    main()
