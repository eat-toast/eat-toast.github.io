
library(XML)
library(data.table)
api_url <- "http://openapi.molit.go.kr:8081/OpenAPI_ToolInstallPackage/service/rest/RTMSOBJSvc/getRTMSDataSvcAptRent"
service_key <- ""

locCode <-c("11110","11140","11170","11200","11215","11230","11260","11290","11305","11320",
            "11350","11380","11410","11440","11470","11500","11530","11545","11560","11590",
            "11620","11650","11680","11710","11740"
            ,'26110','26140','26170','26200','26260','26290','26320','26350','26380','26410','26440','26470'
            ,'26500','26530','267100')

locCode_nm <-c("종로구","중구","용산구","성동구","광진구","동대문구","중랑구","성북구","강북구","도봉구",
               "노원구","은평구","서대문구","마포구","양천구","강서구","구로구","금천구","영등포구","동작구",
               "관악구","서초구","강남구","송파구","강동구"
               ,'부산중구','부산서구','부산동구','부산영도구','부산동래구','부산남구','부산북구','부산해운대구',
               '부산사하구','부산금정구','부산강서구','부산연제구','부산수영구','부산사상구','부산기장군')
year<- as.character(2016:2017)
month<- c(paste0('0', 1:9), '10','11','12')

urllist <- list()
cnt <-0

for(i in 1:length(locCode)){
  for(YYYY in year){
    for(MM in month){
      cnt = cnt + 1
      urllist[cnt] <- paste0(api_url,"?serviceKey=",service_key,'&LAWD_CD=',locCode[i], "&DEAL_YMD=",paste0(YYYY,MM)) 
    }
  }
}

raw.data <- xmlTreeParse(urllist[i], useInternalNodes = TRUE, encoding = "utf-8")
rootNode <- xmlRoot(raw.data)
items <- rootNode[[2]][['items']]


total<-list()
for(i in 1:length(urllist)){
  
  item <- list()
  item_temp_dt<-data.table()
  
  raw.data <- xmlTreeParse(urllist[i], useInternalNodes = TRUE,encoding = "utf-8")
  rootNode <- xmlRoot(raw.data)
  items <- rootNode[[2]][['items']]
  
  if(xmlValue(rootNode[[2]][[4]]) == '0')next() #Total count
  size <- xmlSize(items)
  
  dong_point<- str_locate(urllist[i], 'LAWD_CD')[2]
  dong_point<- str_sub(unlist(urllist[i]), start=(dong_point+2), end = (dong_point+6))
  
  for(j in 1:size){
    item_temp <- xmlSApply(items[[j]],xmlValue)
    item_temp_dt <- data.table( con_year = item_temp[1], #건축년도
                                year = item_temp[2], #년
                                dong = item_temp[3], #법정동
                                bosong = item_temp[4], #보증금액
                                aptnm = item_temp[5], #아파트 이름
                                month = item_temp[6], #월
                                mon_price = item_temp[7], #월세금액
                                day = item_temp[8], #일
                                area = item_temp[9], #전용면적
                                bungi = item_temp[10], #지번
                                code = item_temp[11], #지역코드
                                floor = item_temp[12], #층
                                gu = locCode_nm[which(locCode == dong_point)]
    )
    item[[j]]<-item_temp_dt
  }
  total[[i]] <- rbindlist(item)
  
  print(i)
}
  
apt_data <- rbindlist(total)

library(stingr)
unique(seoul_apt_2017$dong)[str_detect(unique(seoul_apt_2017$dong), '반포')]
idx<- which(seoul_apt_2017$dong == ' 반포동')

idx2<-which(seoul_apt_2017[idx,]$aptnm == '한신15차')
seoul_apt_2017[idx,][idx2,]
