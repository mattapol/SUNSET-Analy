import pandas as pd
import yfinance as yf
import streamlit as st
import datetime as dt
from plotly import graph_objs as go

st.set_page_config(
    page_title="SUNSET - Analysis",
    page_icon="favicon.ico",
)

st.markdown(
    """
    <style>
    .reportview-container {
        background: url("https://color-hex.org/colors/115380.png")
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.sidebar.subheader('SUNSET - Analysis 🌞')

#Retrieving tickers data
ticker_list = pd.read_csv("https://raw.githubusercontent.com/mattapol/SUNSET-Tech/main/csv/symbol_set.csv")
set_th = st.sidebar.selectbox('Choose a Type SET', ('SET50', 'SET100', 'SETHD', 'sSET', 'SETCLMV', 'SETWB'))
if(set_th == 'SET50'):
    ticker_lists = ticker_list[1:51]
elif(set_th == 'SET100'):
    ticker_lists = ticker_list[52:152]
elif(set_th == 'SETHD'):
    ticker_lists = ticker_list[153:183]
elif(set_th == 'sSET'):
    ticker_lists = ticker_list[184:307]
elif(set_th == 'SETCLMV'):
    ticker_lists = ticker_list[308:361]
else:
    ticker_lists = ticker_list[362:392]
    
symbols = ticker_lists['Symbol'].sort_values().tolist()
ticker = st.sidebar.selectbox('Choose a SET Stock 📈', ticker_lists) # Select ticker symbol
infoType = st.sidebar.radio(
        "Choose an info type",
        ('Fundamental', 'Statistics', 'Prediction', 'Short Note')
    )

if(infoType == 'Fundamental'):
    stock = yf.Ticker(ticker)
    info = stock.info 
    st.title('Company Profile 🎢')
    string_logo = '<img src=%s>' % info['logo_url']
    st.markdown(string_logo, unsafe_allow_html=True)
    st.subheader(info['longName']) 
    st.markdown('** Sector **: ' + info['sector'])
    st.markdown('** Industry **: ' + info['industry'])
    st.markdown('** Phone **: ' + info['phone'])
    st.markdown('** Address **: ' + info['address1'] + ', ' + info['city'] + ', ' + info['zip'] + ', '  +  info['country'])
    st.markdown('** Website **: ' + info['website'])
    st.markdown('** Information Summary **')
    st.info(info['longBusinessSummary'])

elif(infoType == 'Statistics'):
    n_days = st.sidebar.number_input("Stock Prices Over Past... (1-365) days📅", 
                                                value=30,
                                                min_value=1, 
                                                max_value=365, 
                                                step=1)
    past_y = n_days * 1 + 1

#show years 
    show_days = int(n_days)
    stock = yf.Ticker(ticker)
    info = stock.info 
    st.title('Statistics 📊')
    st.subheader(info['longName']) 
    st.markdown('** Previous Close **: ' + str(info['previousClose']))
    st.markdown('** Open **: ' + str(info['open']))
    st.markdown('** 52 Week Change **: ' + str(info['52WeekChange']))
    st.markdown('** 52 Week High **: ' + str(info['fiftyTwoWeekHigh']))
    st.markdown('** 52 Week Low **: ' + str(info['fiftyTwoWeekLow']))
    st.markdown('** 200 Week Days **: ' + str(info['twoHundredDayAverage']))

#The Past Of Price Stock 
    start = dt.datetime.today()-dt.timedelta(past_y)
    end = dt.datetime.today()
    df = yf.download(ticker,start,end)
    df = df.reset_index()
    fig = go.Figure(data=[go.Candlestick(x=df['Date'],
                    open=df['Open'], 
                    high=df['High'], 
                    low=df['Low'], 
                    close=df['Close'])])
    st.write('Stock Prices Over Past ', show_days,' Days')
    fig.update_layout(
        title={
            'y':0.9,
            'x':0.5,
            'xanchor': 'center',
            'yanchor': 'top'})
    fig.layout.update(width=800, 
                    height=500, 
                    yaxis_title='Price', 
                    xaxis_title='Date', 
                    xaxis_rangeslider_visible=True)
    fig.update_xaxes(
        rangeslider_visible=True,
        rangeselector=dict(
            buttons=list([
                dict(count=1, label="1m", step="month",                                        
                    stepmode="backward"),
                dict(count=6, label="6m", step="month",  
                    stepmode="backward"),
                dict(count=1, label="YTD", step="year", 
                    stepmode="todate"),
                dict(count=1, label="1y", step="year", 
                    stepmode="backward"),
                dict(step="all")
            ])
        )
    )
    st.plotly_chart(fig, use_container_width=True)

    if n_days < 365 :
        st.success("success 🎉🎊")
    elif n_days == 365 :
        st.success("success 🎉🎊, Limited to the past 1 years")

#Details Stock    
    fundInfo = {
            'Enterprise Value (TH฿)': info['enterpriseValue'],
            'Enterprise To Revenue Ratio': info['enterpriseToRevenue'],
            'Enterprise To Ebitda Ratio': info['enterpriseToEbitda'],
            'Net Income (TH฿)': info['netIncomeToCommon'],
            'Profit Margin Ratio': info['profitMargins'],
            'Forward PE Ratio': info['forwardPE'],
            'PEG Ratio': info['pegRatio'],
            'Price to Book Ratio': info['priceToBook'],
            'Forward EPS (TH฿)': info['forwardEps'],
            'Beta ': info['beta'],
            'Book Value (TH฿)': info['bookValue'],
            'Dividend Rate (%)': info['dividendRate'], 
            'Dividend Yield (%)': info['dividendYield'],
            'Five year Avg Dividend Yield (%)': info['fiveYearAvgDividendYield'],
            'Payout Ratio': info['payoutRatio']
        }
    
    fundDF = pd.DataFrame.from_dict(fundInfo, orient='index')
    fundDF = fundDF.rename(columns={0: 'Value'})
    st.subheader('Stock Info') 
    st.table(fundDF)

#Details Stock  
    marketInfo = {
            "Volume": info['volume'],
            "Average Volume": info['averageVolume'],
            "Market Cap": info["marketCap"],
            "Float Shares": info['floatShares'],
            "Regular Market Price (USD)": info['regularMarketPrice'],
            'Bid Size': info['bidSize'],
            'Ask Size': info['askSize'],
            "Share Short": info['sharesShort'],
            'Short Ratio': info['shortRatio'],
            'Share Outstanding': info['sharesOutstanding']
        }
    
    marketDF = pd.DataFrame(data=marketInfo, index=[0])
    st.table(marketDF)

elif(infoType == 'Prediction'):  
    from datetime import date
    today = date.today()
    START = st.sidebar.date_input("Start date", date(2000, 1, 1))
    TODAY = st.sidebar.date_input("End date", max_value=today)
    st.title("Prediction 📈")

    @st.cache
    def load_data(ticker):
        data = yf.download(ticker, START, TODAY)
        data.reset_index(inplace=True)
        return data

#For Forecast
    n_day = st.sidebar.number_input("Day of prediction... (1-365) days",
                                                value=14,
                                                min_value=1, 
                                                max_value=365, 
                                                step=1)
    period = n_day * 1 + 1

    data_load_state = st.text("Load data... 💫")
    data = load_data(ticker)
    data_load_state.text("Loading data...done! 🎉🎊")

    
    st.subheader('Stock Price 📋')

    def plot_raw_data():
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=data['Date'], y=data['Open'], name='stock_open'))
        fig.add_trace(go.Scatter(x=data['Date'], y=data['Close'], name='stock_close'))
        fig.layout.update(xaxis_rangeslider_visible=True)
        st.plotly_chart(fig)

    plot_raw_data()

    st.write(data.sort_values(by='Date', ascending=False))

    st.subheader('Descriptive Statistics 📚')
    st.write(data.describe())

#Forecasting
    from fbprophet import Prophet
    from fbprophet.plot import plot_plotly
    from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
    import math

    df_train = data[['Date', 'Close']]
    df_train = df_train.rename(columns={"Date": "ds", "Close": "y"})

#Adding the Holiday Effect
    new_year = pd.DataFrame({ #1
        'holiday': 'New Year',
        'ds': pd.to_datetime(['2021-01-01', '2020-01-01', '2019-01-01', '2018-01-01', '2017-01-01', '2016-01-01', '2015-01-01', '2014-01-01', '2013-01-01', '2012-01-01', '2011-01-01', '2010-01-01']),
        'lower_window': 0,
        'upper_window': 1, 
        })
    chinese_new_year = pd.DataFrame({ #2(Special**The First Year)
        'holiday': 'Chinese New Year',
        'ds': pd.to_datetime(['2021-02-12', '2020-02-12', '2019-02-12', '2018-02-12', '2017-02-12', '2016-02-12', '2015-02-12', '2014-02-12', '2013-02-12', '2012-02-12', '2011-02-12', '2010-02-12']),
        'lower_window': 0,
        'upper_window': 1,
        })
    makha_bucha = pd.DataFrame({ #3
        'holiday': 'Makha Bucha',
        'ds': pd.to_datetime(['2021-02-26', '2020-02-26', '2019-02-26', '2018-02-26', '2017-02-26', '2016-02-26', '2015-02-26', '2014-02-26', '2013-02-26', '2012-02-26', '2011-02-26', '2010-02-26']),
        'lower_window': 0,
        'upper_window': 1,
        })
    chakri = pd.DataFrame({ #4
        'holiday': 'Chakri',
        'ds': pd.to_datetime(['2021-04-06', '2020-04-06', '2019-04-06', '2018-04-06', '2017-04-06', '2016-04-06', '2015-04-06', '2014-04-06', '2013-04-06', '2012-04-06', '2011-04-06', '2010-04-06']),
        'lower_window': 0,
        'upper_window': 1,
        })
    songkran = pd.DataFrame({ #5
        'holiday': 'Songkran',
        'ds': pd.to_datetime(['2021-04-13', '2020-04-13', '2019-04-13', '2018-04-13', '2017-04-13', '2016-04-13', '2015-04-13', '2014-04-13', '2013-04-13', '2012-04-13', '2011-04-13', '2010-04-13',
                              '2021-04-14', '2020-04-14', '2019-04-14', '2018-04-14', '2017-04-14', '2016-04-14', '2015-04-14', '2014-04-14', '2013-04-14', '2012-04-14', '2011-04-14', '2010-04-14',
                              '2021-04-15', '2020-04-15', '2019-04-15', '2018-04-15', '2017-04-15', '2016-04-15', '2015-04-15', '2014-04-15', '2013-04-15', '2012-04-15', '2011-04-15', '2010-04-15']),
        'lower_window': 0,
        'upper_window': 1,
        })
    coronation = pd.DataFrame({ #6
        'holiday': 'Coronation',
        'ds': pd.to_datetime(['2021-05-04', '2020-05-04', '2019-05-04', '2018-05-04', '2017-05-04', '2016-05-04', '2015-05-04', '2014-05-04', '2013-05-04', '2012-05-04', '2011-05-04', '2010-05-04']),
        'lower_window': 0,
        'upper_window': 1,
        })
    visakha_bucha = pd.DataFrame({ #7
        'holiday': 'Visakha Bucha',
        'ds': pd.to_datetime(['2021-05-26', '2020-05-26', '2019-05-26', '2018-05-26', '2017-05-26', '2016-05-26', '2015-05-26', '2014-05-26', '2013-05-26', '2012-05-26', '2011-05-26', '2010-05-26']),
        'lower_window': 0,
        'upper_window': 1,
        })
    # queen_suthida_birthday = pd.DataFrame({ #8
    #     'holiday': 'Queen Suthida Birthday',
    #     'ds': pd.to_datetime(['2021-06-03', '2020-06-03', '2019-06-03']),
    #     'lower_window': 0,
    #     'upper_window': 1,
    #     })
    asanha_bucha = pd.DataFrame({ #9
        'holiday': 'Asanha Bucha',
        'ds': pd.to_datetime(['2021-07-24', '2020-07-24', '2019-07-24', '2018-07-24', '2017-07-24', '2016-07-24', '2015-07-24', '2014-07-24', '2013-07-24', '2012-07-24', '2011-07-24', '2010-07-24']),
        'lower_window': 0,
        'upper_window': 1,
        })
    buddhist_lent = pd.DataFrame({ #10
        'holiday': 'Buddhist Lent',
        'ds': pd.to_datetime(['2021-07-25', '2020-07-25', '2019-07-25', '2018-07-25', '2017-07-25', '2016-07-25', '2015-07-25', '2014-07-25', '2013-07-25', '2012-07-25', '2011-07-25', '2010-07-25']),
        'lower_window': 0,
        'upper_window': 1,
        })
    s_asanha_bucha = pd.DataFrame({ #11(Special**Only Year-Substitution for Asanha Bucha Day)
        'holiday': 'Substitution for Asanha Bucha',
        'ds': pd.to_datetime(['2021-07-26', '2020-07-26', '2019-07-26', '2018-07-26', '2017-07-26', '2016-07-26', '2015-07-26', '2014-07-26', '2013-07-26', '2012-07-26', '2011-07-26', '2010-07-26']),
        'lower_window': 0,
        'upper_window': 1,
        })
    # king_maha_vajiralongkorn_birthday = pd.DataFrame({ #12
    #     'holiday': 'King Maha Vajiralongkorn Birthday',
    #     'ds': pd.to_datetime(['2021-07-28', '2020-07-28', '2019-07-28']),
    #     'lower_window': 0,
    #     'upper_window': 1,
    #     })
    mother_birthday = pd.DataFrame({ #13
        'holiday': 'Queen Sirikit, Mother’s Birthday',
        'ds': pd.to_datetime(['2021-08-12', '2020-08-12', '2019-08-12', '2018-08-12', '2017-08-12', '2016-08-12', '2015-08-12', '2014-08-12', '2013-08-12', '2012-08-12', '2011-08-12', '2010-08-12']),
        'lower_window': 0,
        'upper_window': 1,
        })
    prince_of_songkla_memorial = pd.DataFrame({ #14 
        'holiday': 'Prince of Songkla Memorial',
        'ds': pd.to_datetime(['2021-09-24', '2020-09-24', '2019-09-24', '2018-09-24', '2017-09-24', '2016-09-24', '2015-09-24', '2014-09-24', '2013-09-24', '2012-09-24', '2011-09-24', '2010-09-24']),
        'lower_window': 0,
        'upper_window': 1,
        })
    # king_bhumibol_the_great_memorial = pd.DataFrame({ #15
    #     'holiday': 'King Bhumibol Adulyadej The Great Memorial',
    #     'ds': pd.to_datetime(['2021-10-13', '2020-10-13', '2019-10-13', '2018-10-13', '2017-10-13']),
    #     'lower_window': 0,
    #     'upper_window': 1,
    #     })
    # s_king_chulalongkorn_memorial = pd.DataFrame({ #16(Special**Only Year-Substitution for King Chulalongkorn Memorial Day (Shifted from 25th Oct))
    #     'holiday': 'Substitution for King Chulalongkorn Memorial', 
    #     'ds': pd.to_datetime(['2021-10-22']),
    #     'lower_window': 0,
    #     'upper_window': 1,
    #     })
    king_chulalongkorn_memorial = pd.DataFrame({ #17
        'holiday': 'king_chulalongkorn_memorial',
        'ds': pd.to_datetime(['2021-10-23', '2020-10-23', '2019-10-23', '2018-10-23', '2017-10-23', '2016-10-23', '2015-10-23', '2014-10-23', '2013-10-23', '2012-10-23', '2011-10-23', '2010-10-23']),
        'lower_window': 0,
        'upper_window': 1,
        })
    king_bhumibol_the_great_birthday = pd.DataFrame({ #18
        'holiday': 'King Bhumibol The Great Birthday',
        'ds': pd.to_datetime(['2021-12-05', '2020-12-05', '2019-12-05', '2018-12-05', '2017-12-05', '2016-12-05', '2015-12-05', '2014-12-05', '2013-12-05', '2012-12-05', '2011-12-05', '2010-12-05']),
        'lower_window': 0,
        'upper_window': 1,
        })
    # s_king_bhumibol_the_great_birthday = pd.DataFrame({ #19(Special**Only Year-Substitution for King Bhumibol The Great Birthday)
    #     'holiday': 'Substitution for King Bhumibol The Great Birthday',
    #     'ds': pd.to_datetime(['2021-12-06']),
    #     'lower_window': 0,
    #     'upper_window': 1,
    #     })
    constitution = pd.DataFrame({ #20
        'holiday': 'Constitution',
        'ds': pd.to_datetime(['2021-12-10', '2020-12-10', '2019-12-10', '2018-12-10', '2017-12-10', '2016-12-10', '2015-12-10', '2014-12-10', '2013-12-10', '2012-12-10', '2011-12-10', '2010-12-10']),
        'lower_window': 0,
        'upper_window': 1,
        })
    new_year_eve = pd.DataFrame({ #21
        'holiday': 'New Year Eve',
        'ds': pd.to_datetime(['2021-12-31', '2020-12-31', '2019-12-31', '2018-12-31', '2017-12-31', '2016-12-31', '2015-12-31', '2014-12-31', '2013-12-31', '2012-12-31', '2011-12-31', '2010-12-31']),
        'lower_window': 0,
        'upper_window': 1,
        })
    new_holidays = pd.concat((new_year, chinese_new_year, makha_bucha, chakri, songkran, coronation, visakha_bucha, asanha_bucha, buddhist_lent, s_asanha_bucha, 
                               mother_birthday, prince_of_songkla_memorial, king_chulalongkorn_memorial, king_bhumibol_the_great_birthday, constitution, new_year_eve)) #queen_suthida_birthday, king_bhumibol_the_great_memorial, s_king_chulalongkorn_memorial, king_maha_vajiralongkorn_birthday, s_king_bhumibol_the_great_birthday

#Make Prediction
    m1 = Prophet(holidays=new_holidays, daily_seasonality=True)
    m1.fit(df_train)
    # Create Future Dates
    future = m1.make_future_dataframe(periods=period, include_history=True)
    # Predict Prices
    forecast = m1.predict(future)

    st.subheader('Plot Chart Prediction')
    fig1 = plot_plotly(m1, forecast)
    st.plotly_chart(fig1)

    #Sklearn Metrics
    metric_df = (forecast.set_index('ds')[['yhat']].join(df_train.set_index('ds')[['y']]).reset_index()).sort_values(by='ds', ascending=False).rename(columns={"ds": "Date", "yhat": "Predicted", "y": "Actual"})
    metric_df.dropna(inplace=True)
    
    #R2 OR R-Squared
    st.sidebar.write('R-Squared')
    st.sidebar.write(r2_score(metric_df.Actual, metric_df.Predicted))
    #MAE OR Mean Absolute Error
    st.sidebar.write('Mean Absolute Error')
    st.sidebar.write(mean_absolute_error(metric_df.Actual, metric_df.Predicted))
    #MSE OR Mean Squared Error
    st.sidebar.write('Mean Squared Error')
    st.sidebar.write(mean_squared_error(metric_df.Actual, metric_df.Predicted))
    #RMSE OR Root Mean Square Error
    st.sidebar.write('Root Mean Square Error')
    st.sidebar.write(math.sqrt(mean_squared_error(metric_df.Actual, metric_df.Predicted)))

    st.subheader('Price Forecast 💸')
    st.markdown('Date = **_วันที่หุ้นเปิดทําการ_**')
    st.markdown('Predic_lower = **_ราคาทํานายหุ้นขอบเขตตํ่าสุด_**')
    st.markdown('Predic_upper = **_ราคาทํานายหุ้นขอบเขตสูงสุด_**')
    st.markdown('Predicted = **_ราคาทํานายหุ้น_**')
    st.markdown('Actual = **_ราคาหุ้นจริง_**')
    st.caption('**** <NA> <ราคาหุ้นค่าจริงที่ยังไม่ออก>****')

    # forecast.drop(['trend', 'trend_lower', 'trend_upper', 'Asanha Bucha', 'Asanha Bucha_lower', 'Asanha Bucha_upper',
    #                'Buddhist Lent', 'Buddhist Lent_lower', 'Buddhist Lent_upper', 'Chakri', 'Chakri_lower', 'Chakri_upper', 
    #                'Chinese New Year', 'Chinese New Year_lower', 'Chinese New Year_upper', 'Constitution', 'Constitution_lower',
    #                'Constitution_upper', 'Coronation', 'Coronation_lower', 'Coronation_upper', 'King Bhumibol The Great Birthday',
    #                'King Bhumibol The Great Birthday_lower', 'King Bhumibol The Great Birthday_upper', 'Makha Bucha', 
    #                'Makha Bucha_lower', 'Makha Bucha_upper', 'New Year', 'New Year_lower', 'New Year_upper', 'New Year Eve',
    #                'New Year Eve_lower', 'New Year Eve_upper', 'Prince of Songkla Memorial', 'Prince of Songkla Memorial_lower',
    #                'Prince of Songkla Memorial_upper', 'Queen Sirikit, Mother’s Birthday', 'Queen Sirikit, Mother’s Birthday_lower',
    #                'Queen Sirikit, Mother’s Birthday_upper', 'Songkran', 'Songkran_lower', 'Songkran_upper', 'Substitution for Asanha Bucha',
    #                'Substitution for Asanha Bucha_lower', 'Substitution for Asanha Bucha_upper', 'Visakha Bucha', 'Visakha Bucha_lower',
    #                'Visakha Bucha_upper', 'additive_terms', 'additive_terms_lower', 'additive_terms_upper', 'daily', 'daily_lower',
    #                'daily_upper', 'holidays', 'holidays_lower', 'holidays_upper', 'king_chulalongkorn_memorial',
    #                'king_chulalongkorn_memorial_lower', 'king_chulalongkorn_memorial_upper', 'weekly', 'weekly_lower', 'weekly_upper',
    #                'yearly', 'yearly_lower', 'yearly_upper', 'multiplicative_terms', 'multiplicative_terms_lower', 'multiplicative_terms_upper'], axis=1, inplace=True)
    # fc = forecast.sort_values(by='ds', ascending=False).rename(columns={"ds": "Date", "yhat_lower": "Predic_lower", "yhat_upper": "Predic_upper", "yhat": "Predicted", "y": "Actual"})
    # st.write(fc)

    mt = (forecast.set_index('ds')[['yhat', 'yhat_lower', 'yhat_upper']].join(df_train.set_index('ds')[['y']]).reset_index()).sort_values(by='ds', ascending=False).rename(columns={"ds": "Date", "yhat_lower": "Predic_lower", "yhat_upper": "Predic_upper", "yhat": "Predicted", "y": "Actual"})
    st.write(mt)

    st.subheader('Forecast Components')
    fig1 = m1.plot_components(forecast)
    st.write(fig1)

else:
    def show():
        st.title('✅ Short Note')
        # Define initial state.
        if "todos" not in st.session_state:
            st.session_state.todos = [
                {"description": "Delete", "done": True},
                {
                    "description": "Test 🕹",
                    "done": False,
                },
            ]

        # Define callback when text_input changed.
        def new_todo_changed():
            if st.session_state.new_todo:
                st.session_state.todos.append(
                    {
                        "description": st.session_state.new_todo,
                        "done": False,
                    }
                )

        # Show widgets to add new TODO.
        st.write(
            "<style>.main * div.row-widget.stRadio > div{flex-direction:row;}</style>",
                    unsafe_allow_html=True,
        )
        st.sidebar.text_input("What do you need to remember?", on_change=new_todo_changed, key="new_todo")
                
        # Show all TODOs.
        write_todo_list(st.session_state.todos)

    def write_todo_list(todos):
        "Display the todo list (mostly layout stuff, no state)."
        st.sidebar.write("")
        col1, col2, _ = st.columns([0.05, 0.8, 0.15])
        all_done = True
        for i, todo in enumerate(todos):
            done = col1.checkbox("", todo["done"], key=str(i))
            if done:
                format_str = (
                        '<span style="color: grey; text-decoration: line-through;">{}</span>'
                )
            else:
                format_str = "{}"
                all_done = False
            col2.markdown(
                format_str.format(todo["description"]),
                unsafe_allow_html=True,
            )
            
        if all_done:
            st.success("Nice job on finishing all NOTE items! Good Luck 🎇")

    if __name__ == "__main__":
            show()
