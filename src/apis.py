import requests
import base64
import pandas as pd
import json
from pandas.io.json import json_normalize

def api_centros(url):
    colegios = url
    r = requests.get(colegios)

    data = r.json()

    with open('../input/colegios.csv', 'w') as fp:
        json.dump('../input/colegios.csv', fp)

    colegios = data['@graph']
    df = json_normalize(colegios)

    df.drop(['@id', '@type', 'title', 'id', 'relation', 'address.district.@id',
             'address.area.@id', 'organization.organization-desc',
             'organization.schedule', 'organization.accesibility', 'organization.services']
            , axis=1)
    #df.to_excel("centros_educativos_final.xls")
    df.to_excel("../input/centros_educativos_final.xls")
    return df


def import_excel(url):
    resp = requests.get(url)
    output = open('../input/excel_file_.xls', 'wb')
    output.write(resp.content)
    output.close()
    return output

def api_idealista(api_key,secret_key,file_path):
    API_KEY = api_key
    SECRET_KEY = secret_key
    message = API_KEY + ":" + SECRET_KEY
    message_bytes = message.encode('ascii')
    auth = "Basic " + str(base64.b64encode(message_bytes)).replace("b'", "").replace("'", "")
    def get_oauth_token(auth):
        headers = {"Authorization": auth, "Content-Type": "application/x-www-form-urlencoded"}
        params_dic = {"grant_type": "client_credentials",
                      "scope": "read"}
        response = requests.post("https://api.idealista.com/oauth/token", headers=headers, params=params_dic)
        print("Connection Response:", response)
        bearer_token = json.loads(response.text)['access_token']
        return bearer_token

    def search_api(token, url):
        headers = {'Content-Type': 'Content-Type: multipart/form-data;', 'Authorization': 'Bearer ' + token}
        content = requests.post(url, headers=headers)
        print("Content Response:",content)
        result = content.json()
        return result

    country = 'es'  # values: es, it, pt
    language = 'es'  #
    max_items = '50'
    operation = 'sale'
    property_type = 'homes'
    order = 'publicationDate'
    center = '40.4146500,-3.7004000'
    distance = '10000'
    sort = 'desc'

    df_tot = pd.read_csv(file_path)
    lista_codigos_totales = df_tot["propertyCode"]

    limit = 1
    repetidos = False
    i = 1

    while repetidos == False:
    #for i in range(limit):
        url = ('https://api.idealista.com/3.5/' + country +
               '/search?operation=' + operation +
               '&maxItems=' + max_items +
               '&order=' + order +
               '&center=' + center +
               '&distance=' + distance +
               '&propertyType=' + property_type +
               '&sort=' + sort +
               '&numPage=%s' +
               '&language=' + language) % (i)

        a = search_api(get_oauth_token(auth), url)

        for h in a['elementList']:
            if int(h["propertyCode"]) in lista_codigos_totales:
                print("repetidos paramos las requests")
                repetidos = True
        i += 1

        df = pd.DataFrame.from_dict(a['elementList'])
        df_tot = pd.concat([df_tot, df])
        df_tot.drop_duplicates(subset="propertyCode",
                                 keep=False, inplace=True)

    print("DONE")

    df_tot.to_csv(file_path, index=False)
    return df_tot

