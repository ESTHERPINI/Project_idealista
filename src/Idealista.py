import base64
import requests
import json

API_KEY="n421au7se73p0n67unk7u1h0jminlna9"
SECRET_KEY="lm2fMZK1dG0H"
message=API_KEY+":"+SECRET_KEY
print(message)

message_bytes=message.encode("ascii")
print(message_bytes)

base64_bytes="Basic"+str(base64.b64encode(message_bytes))
print(base64_bytes)

#Autorization key base 64

#diccionario token
def get_oauth_token():
    headers = {"Authorization":"Basic bjQyMWF1N3NlNzNwMG42N3Vuazd1MWgwam1pbmxuYTk6bG0yZk1aSzFkRzBI",
               "Content-Type":"application/x-www-form-urlencoded"}
    params_dic = {"grant_type" : "client_credentials","scope" : "read"}
    response = requests.post("https://api.idealista.com/oauth/token", headers=headers, params = params_dic)
    print(response)
    bearer_token = json.loads(response.text)['access_token']
    return bearer_token

def search_api(token, url):
    headers = {'Content-Type': 'Content-Type: multipart/form-data;', 'Authorization' : 'Bearer ' + token}
    content = requests.post(url, headers = headers)
    result = content.json()
    return result


if __name__=="__main__":
    get_oauth_token()
    