import streamlit as st
import plotly.express as px
import joblib
import pandas as pd
import numpy as np


def preprocess_data_(data):
    #AGE
    if data[7] < 35:
        data.append(0)
        data.append(1)
    elif 35 <= data[7] <= 55:
        data.append(0)
        data.append(0)
    else:
        data.append(1)
        data.append(0)

    #BMI
    if data[5] < 18.5:
        data.append(0)
        data.append(0)
        data.append(0)
    elif 18.5 <= data[5] < 24.9:
        data.append(1)
        data.append(0)
        data.append(0)
    elif 24.9 <= data[5] < 29.9:
        data.append(0)
        data.append(1)
        data.append(0)
    else:
        data.append(0)
        data.append(0)
        data.append(1)

    #GLUCOSE
    if data[1] < 140:
        data.append(0)
        data.append(0)
    elif 140 <= data[1] < 200:
        data.append(0)
        data.append(1)
    else:
        data.append(1)
        data.append(0)

    #BLOODPRESSURE
    if data[2] < 79:
        data.append(0)
        data.append(0)
    elif 79 <= data[2] < 89:
        data.append(0)
        data.append(1)
    else:
        data.append(1)
        data.append(0)

    #INSULIN
    if 16 <= data[4] <= 166:
        data.append(1)
    else:
        data.append(0)
        # Burayı böyle doldurmalısın, data[4] tamamen rastgele yazdığım bir şey

    return data


st.set_page_config(layout="wide",page_title="👨🏻‍⚕️ Diyabet Tahmini")


data = pd.read_csv('database/diabetes.csv')



@st.cache_data
def get_data():
    df = pd.read_csv('database/diabetes.csv')
    return df


def get_model():
    model = joblib.load("streamlit/streamlit/pipeline/model.joblib")
    return model


st.header("👨🏻‍⚕️ Diyabet Tahmini ")
tab_home, tab_info, tab_modelinfo,  tab_model = st.tabs(["Ana Sayfa", "Hakkımızda Bilgi", "Model Hakkında Bilgi", "Model"])

# TAB HOME
column_diabetes, column_dataset = tab_home.columns(2)

column_diabetes.subheader("Diyabet Nedir?")
column_diabetes.markdown(
    "Halk arasında genel olarak  şeker hastalığı olarak tabir edilen Diabetes Mellitus, genel olarak kanda glukoz (şeker) seviyesinin normalin üzerine çıkması,"
    " buna bağlı olarak normalde şeker içermemesi gereken idrarda şekere rastlanmasıdır. Farklı türevleri bulunan diyabet hastalığı, ülkemizde ve dünyada en sık"
    " rastlanan hastalıklar arasında yer alır. Uluslararası Diyabet Federasyonu'nun sağlamış olduğu istatistiki verilere göre her 11 yetişkinden biri diyabet "
    "hastalığına sahip olmakla birlikte her 6 saniyede 1 birey diyabet kaynaklı sorunlar nedeniyle hayatını kaybetmektedir.")
column_diabetes.subheader(" Kolonlar :")
column_diabetes.markdown("* Pregnancies = Hamile kalma sayısı (0-17)")
column_diabetes.markdown("* Glucose = Kan glukoz Değeri (0-199)")
column_diabetes.markdown("* Blood Pressure = Kan Basıncı (0-122 mm/Hg)")
column_diabetes.markdown("* Skin Thickness = Deri Kalınlığı (0-99 mm)")
column_diabetes.markdown("* Insulin = İnsulin Değeri (0-846 mu U/ml)")
column_diabetes.markdown("* BMI = Vücut Kitle İndeksi kg ve m^2")
column_diabetes.markdown(
    "* Diabetes Pedigree Function = Kişinin şeker hastalığına genetik olarak yatkınlığı (0.078-2.42)")
column_diabetes.markdown("* Age = Yas Değeri")
column_diabetes.markdown("* Outcome = Çıktı (0 / 1)")

column_dataset.subheader("Diabetes Veri Seti")
df = get_data()
column_dataset.dataframe(df)
# st.checkbox("Use container width", value=False, key="use_container_width")
# column_dataset.dataframe(get_data(), use_container_width = st.session_state.use_container_width)

# TAB INFO
column_Tugba, column_Temel = tab_info.columns(2)
column_Tugba.image('streamlit/streamlit/galeri/tugba_pp.png', width=300)
column_Tugba.subheader("Tuğba AKTAŞ")
column_Tugba.markdown("Merhaba ben Tuğba Aktaş, Veri Bilimiyle ilgili araştırmalar yapıyor ve bu alanda mentor yardımcılığı yapıyorum. "
                      "Yani yapıyorum herhalde. Tek hayalim işsiz kalmamak.")
column_Tugba.markdown("LinkedIn : [TUAKTAS_LinkedIn](https://www.linkedin.com/in/tugbaaktas/)")
column_Tugba.markdown("Github : [TUAKTAS_Github](https://github.com/tubaaktas)", )


column_Temel.image('streamlit/streamlit/galeri/temelinko.png', width=300)
column_Temel.subheader("İsmail Mert TEMEL")
column_Temel.markdown("Merhaba ben İsmail Mert TEMEL, Makine Öğrenmesi ve Veri Bilimi üzerine ufak çalışmalar yapmaya çalışıyorum. Mezun olup apartman yöneticisi olmak kariyerim adına en büyük planım.")
column_Temel.markdown("LinkedIn : [TEMELINKO_LinkedIn](https://www.linkedin.com/in/ismail-mert-temel-688abb198/)",unsafe_allow_html=True)
column_Temel.markdown("Github : [TEMELINKO_Github](https://github.com/Temelinko)")


# TAB MODEL İNFO

tab_modelinfo.subheader("Model İle İlgili Neler Yaptık")
tab_modelinfo.markdown(" Kaggledan temin ettiğimiz diyabet veri seti ile başarılı tahminler "
                       " üretebilen bir makine öğrenmesi algoritması kurmayı amaçladık."
                       " Bu süreçte literatür taraması ile işe başladık. Check_df adlı bir fonksiyon oluşturarak"
                       " veriyi inceledik. BoxPlot kullanarak aykırı değer analizi yaptık."
                       " Sütunlarımız arasında bulunan ilişkileri Korelasyon Grafiği kullanarak ortaya çıkarttık."
                       " Veri setinde hiç boş değer gözükmüyor olmasına rağmen bazı değerlerin 0 olamayacağından bu değerlerin boş değer"
                       " olduğu kanısına vardık. Veri seti normal dağılım göstermediğinden boş değerlerimizi medyan kullanarak doldurduk. ")
tab_modelinfo.markdown(" (Opsiyonel olarak veri seti 'Outcome' değeri 0 olan 500 girdinin sadece 250 girdiyi de kullanarak normal dağılım elde edilebilirdi.)")

column_left, column_middle, column_right = tab_modelinfo.columns(3)
column_left.image('streamlit/streamlit/galeri/DataCheck.jpeg', width=454)
column_left.markdown("* check_df(data) Fonksiyonu Oluşan Çıktılar: Shape, Null Analizi, Veri Türü")

column_middle.image('streamlit/streamlit/galeri/myplot.jpeg', use_column_width=True)
column_middle.markdown("* BoxPlot Grafiği")

column_right.image('streamlit/streamlit/galeri/myplot_corr.jpeg', use_column_width=True)
column_right.markdown("* Korelasyon Grafiği")

tab_modelinfo.markdown(" Veri setine hiç bir işlem uygulamadan Base Modelimizi kurduk. Bunun amacı veri setinde değişiklik "
                       " yapmadan önce ve sonrasında model değerlendirme metriklerini karşılaştırabilmektir."
                       " Aykırı degerlerin baskılanması ve boş değerlerin doldurulma işleminden sonra öznitelik çıkarımı"
                       " işlemlerini yaptık. Temel amacımız makine öğrenmesi işlemini daha iyi yapabilmekti."
                       " Aşırı öğrenmeden kaçınmak adına, yaptığımız özellik çıkarımını belli düzeyde kestik.")
column_left1, column_right1 = tab_modelinfo.columns(2)
column_left1.image('streamlit/streamlit/galeri/accuracy_feoncesi.jpeg', use_column_width=True)
column_left1.markdown("* Accuracy Değeri Öznitelik Çıkarımı Öncesi")

column_right1.image('streamlit/streamlit/galeri/accuracy_fesonrasi.jpeg', use_column_width=True)
column_right1.markdown("* Accuracy Değeri Öznitelik Çıkarımı Sonrası")

column_left1.image('streamlit/streamlit/galeri/f1_feoncesi.jpeg', use_column_width=True)
column_left1.markdown("* F1 Değeri Öznitelik Çıkarımı Öncesi")

column_right1.image('streamlit/streamlit/galeri/f1_fesonrasi.jpeg', use_column_width=True)
column_right1.markdown("* F1 Değeri Öznitelik Çıkarımı Sonrası")

tab_modelinfo.markdown(" Son olarak Final Modeller oluşturup değerlendirme metriklerine göre"
                       " hangi modelin daha iyi olabileceğini inceledik. "
                       " Bu işlemi yaparken f1 ve accuracy değerlerine baktık. LightGBM, RF, XGBoost modellerinin çok iyi performans"
                       " verdiğini gördük ve bu modelleri birleştirebilmek ve sonuçları daha iyi etkileyeceğini"
                       " düşünerek Voting Classifieri kullandık. ")
tab_modelinfo.subheader("Voting Calssifier Nedir? ")
tab_modelinfo.markdown(" Modellerin karar verme yeteneklerini metriklere göre birleştirebilen"
                       " ve daha iyi metrikler ortaya"
                       " sunan bir yöntemdir. Bunu yaparken kullandığınız modellerin target değerine oylar veren"
                       " bir kurul olarak düşünebiliriz ya oy birliği ile ya da oy coğunluğu ile target değerini "
                       " belirleyen ve genel olarak modelin performans metriğini iyileştiren bir yöntemdir.")
tab_modelinfo.markdown(" Normalde bu yöntemi kullanmayı planlıyorduk fakat yasadığımız teknik aksaklıklar sebebi ile "
                       " bundan vazgeçtik. Final modeli olarak Random Forest kullanmaya karar verdik.")

column_left2, column_right2 = tab_modelinfo.columns(2)
column_left2.image('streamlit/streamlit/galeri/RFsonuc.jpeg', width=780)
column_left2.markdown("* RF (Random Forest) Model Sonuçları")

column_right2.image('streamlit/streamlit/galeri/ConfusionMatrix.jpeg', width=780)
column_right2.markdown("* RF (Random Forest) Confusion Matrix")


tab_modelinfo.markdown("En sonunda ise bu projeyi size sunabilmek adına StreamLit ile bir arayüz tasarladık. "
                       "Tasarladiğımız arayüz gayet basit olup direkt kullanıcı dostu bir arayüz olmasına özen "
                       "gösterdik.")

# TAB MODEL

model = get_model()

column_model, column_modelresult = tab_model.columns(2)
#Pregnancies,Glucose,BloodPressure,SkinThickness,Insulin,BMI,DiabetesPedigreeFunction,Age
Pregnancies = column_model.number_input("Hamilelik sayınız",min_value=0,max_value=20)
Glucose = column_model.number_input("Glikoz değeriniz",min_value=1,max_value=400)
BloodPressure = column_model.number_input("Kan Basıncınız",min_value=1, max_value=150)
SkinThickness = column_model.number_input("Deri Kalınlığınız",min_value=0.01)
Insulin = column_model.number_input("İnsulin değeriniz",min_value=10,max_value=400)
BMI = column_model.number_input("BMI değeriniz",min_value=5,max_value=150)
DiabetesPedigreeFunction = column_model.number_input("Diyabet Soy Ağacı değeriniz",min_value=0)
Age = column_model.slider("Yaşınız",min_value=5, max_value=100)


tahmin = preprocess_data_([Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age])



if column_model.button("Tahmin Etmek İçin Basınız"):
    prediction = model.predict(np.array(tahmin).reshape(1, -1))[0]
    # tab_model.success(f"tahmın edılen deger: {prediction[0]}")
    if prediction == 1:
        column_modelresult.image('streamlit/streamlit/galeri/diyabetsiniz.png', width=600)
    else:
        column_modelresult.image('streamlit/streamlit/galeri/diyabetdegilsiniz.png', width=600)
        column_modelresult.success("Diyabet Değilsiniz")
        column_modelresult.balloons()

