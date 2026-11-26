
import joblib
import nltk
import streamlit as st
from nltk.corpus import stopwords, wordnet
from nltk.sentiment import SentimentIntensityAnalyzer
from nltk.stem import WordNetLemmatizer




@st.cache_resource
def tai_du_lieu_nltk():
    nltk.download('punkt_tab')
    nltk.download('stopwords')
    nltk.download('wordnet')
    nltk.download('omw-1.4')
    nltk.download('averaged_perceptron_tagger_eng')
    nltk.download('vader_lexicon')


@st.cache_resource
def nap_mo_hinh():
    return joblib.load('model.pkl')   


tai_du_lieu_nltk()
hop = nap_mo_hinh()
vader = SentimentIntensityAnalyzer()
lemmatizer = WordNetLemmatizer()
STOPWORDS = set(stopwords.words('english'))



TU_PHU_DINH = {'not', 'no', 'nor', 'never', "n't", 'don', 'didn', 'doesn', 'isn', 'wasn',
               'aren', 'weren', 'won', 'wouldn', 'couldn', 'shouldn', 'hasn', 'haven'}


def doi_nhan_tu_loai(nhan):
    if nhan.startswith('J'):
        return wordnet.ADJ
    if nhan.startswith('V'):
        return wordnet.VERB
    if nhan.startswith('R'):
        return wordnet.ADV
    return wordnet.NOUN


def tien_xu_ly(van_ban, giu_tu_phu_dinh=False):
    cac_tu = nltk.word_tokenize(van_ban.lower())
    cac_tu = [tu for tu in cac_tu if tu.isalpha() or tu == "n't"]
    tu_da_xu_ly = []
    for tu, nhan in nltk.pos_tag(cac_tu):
        tu_goc = lemmatizer.lemmatize(tu, doi_nhan_tu_loai(nhan))
        if tu_goc == "n't":
            tu_goc = 'not'
        tu_da_xu_ly.append(tu_goc)
    if giu_tu_phu_dinh:
        tu_can_bo = STOPWORDS - TU_PHU_DINH
    else:
        tu_can_bo = STOPWORDS
    return [tu for tu in tu_da_xu_ly if tu not in tu_can_bo]




def du_doan_naive_bayes(cau):
  
    cac_tu = tien_xu_ly(cau, hop['giu_tu_phu_dinh'])
    tu_trong_cau = set(cac_tu)
    cap_trong_cau = set(nltk.bigrams(cac_tu))


    dac_trung = {}
    for tu in hop['tu_vung']:
        if tu in tu_trong_cau:
            dac_trung[tu] = True
    for cap in hop['cap_tu_vung']:
        if cap in cap_trong_cau:
            dac_trung[cap[0] + ' ' + cap[1]] = True

    xac_suat = hop['mo_hinh'].prob_classify(dac_trung)
    nhan = xac_suat.max()
    return nhan, xac_suat.prob(nhan), cac_tu, list(dac_trung)


def du_doan_vader(cau):
    diem = vader.polarity_scores(cau)
    if diem['compound'] >= 0:
        return 'pos', diem
    return 'neg', diem


def hien_nhan(nhan):
    if nhan == 'pos':
        st.success('😊 TÍCH CỰC (khen)')
    else:
        st.error('😞 TIÊU CỰC (chê)')



st.set_page_config(page_title='Phân tích cảm xúc đánh giá phim', page_icon='🎬')
st.title(' Phân tích cảm xúc đánh giá phim')
st.caption('Mô hình: Naive Bayes (' + hop['ten_cau_hinh'] + ') | '
           'Độ chính xác trên tập thi: ' + str(round(hop['do_chinh_xac'] * 100, 2)) + '%')

CAU_VI_DU = {
    '(Tự nhập)': '',
    'Ví dụ tích cực': 'An absolutely wonderful film with brilliant performances. Highly recommended!',
    'Ví dụ tiêu cực': 'The plot was predictable and the actors were awful. A boring waste of time.',
    'Ví dụ khó (mỉa mai)': 'Oh great, another two hours of my life I will never get back.',
}
lua_chon = st.selectbox('Chọn câu ví dụ hoặc tự nhập:', list(CAU_VI_DU))
cau_nhap = st.text_area('Nhập bài đánh giá phim (tiếng Anh):', value=CAU_VI_DU[lua_chon], height=150)

if st.button(' Phân tích', type='primary'):        
    if cau_nhap.strip() == '':
        st.warning('Bạn chưa nhập nội dung.')
    else:
        nhan_nb, tin_cay, cac_tu, tu_nhan_ra = du_doan_naive_bayes(cau_nhap)
        nhan_vader, diem_vader = du_doan_vader(cau_nhap)

        cot_trai, cot_phai = st.columns(2)          
        with cot_trai:
            st.subheader('Naive Bayes')
            hien_nhan(nhan_nb)
            st.metric('Độ tin cậy', str(round(tin_cay * 100, 1)) + '%')
            st.progress(float(tin_cay))
        with cot_phai:
            st.subheader('VADER (để đối chiếu)')
            hien_nhan(nhan_vader)
            st.metric('Điểm compound', round(diem_vader['compound'], 3))

        with st.expander('Xem chi tiết'):
            st.write('**Các từ sau tiền xử lý:**', cac_tu)
            st.write('**Các từ mô hình nhận ra:**', tu_nhan_ra if tu_nhan_ra else 'Không có từ nào')