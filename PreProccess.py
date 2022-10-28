
import nltk


def delete_Stopword(txt):
    file = nltk.word_tokenize(txt)
    newFile=''
    stopWord=[]
    fin=open('data/stopword.txt',encoding='utf8')

    for word in fin.readlines():
        stopWord.append(word.replace('\n', '').lower().replace('\ufeff', '').lower().replace('\ufeff', '').upper().replace('ك' , 'ک').replace(" " ,""))



    for word in file:
        word=word.replace(' ', '')
        if word in stopWord:
            continue
        else:
            newFile = newFile + ' ' + word


    return newFile




def normalizer(txt):
    file = nltk.word_tokenize(txt)
    newfile=''
    for word in file:
        word = word.replace(" ", "").replace("\u200c", "").replace(".", ". ").upper().replace("\ufeff\n","").lower().replace(
            "،", "، ").upper().replace('ة' , 'ه').replace('ي', 'ی').replace("؛" , '؛ ').upper().replace("؛" , ' ؛').lower().replace("." , ' .').lower().replace(
            "،", " ،").lower().replace('\xa0','').replace('ک' , 'ك').replace('پ' , 'ب').replace('گ' ,'ك' )
        newfile = newfile + ' ' + word

    return newfile




