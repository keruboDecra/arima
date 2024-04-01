import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
import streamlit as st

# Function to load data
def load_data():
    data = {
        'Metrics': ['Enrolled learners', 'Active Learners', 'Learner drop outs', 'Graduated learners', '# Fellows', 'Staff count', '%African Staff',
                    'Offer Acceptance Rate', 'Cash in bank', 'Monthly Recurring Revenue (MRR)', 'Revenue Booked', 'Cost per hire'],
        'May_Week1': [35890, 28000, 0, 10000, 34, 500, 0.5, 100, 80000000, 220000, 120000, 4000],
        'May_Week2': [35890, 27900, 100, 10000, 40, 498, 0.5, 100, 85000000, 225000, 130000, 4000],
        'May_Week3': [35890, 27500, 400, 10000, 41, 507, 0.48, 100, 85900000, 215000, 115000, 4000],
        'May_Week4': [35890, 27000, 500, 10000, 50, 507, 0.48, 100, 80500000, 230000, 115000, 4000],
        'June_Week1': [35890, 27000, 0, 10000, 56, 507, 0.48, 100, 82500000, 230000, 135000, 5200],
        'June_Week2': [35890, 26900, 100, 10000, 61, 505, 0.49, 100, 82500000, 235000, 140000, 5200],
        'June_Week3': [35890, 26500, 400, 10000, 63, 504, 0.489, 100, 82500000, 250000, 137000, 5200],
        'June_Week4': [35890, 26400, 100, 36400, 67, 506, 0.5, 99, 82500000, 250000, 145000, 4900],
        'July_Week1': [40000, 34000, 0, 0, 80, 506, 0.5, 99, 85000000, None, 142000, 5200],
        'July_Week2': [42500, 32000, 2000, 560, 120, 510, 0.489, 99, 85200000, None, 150000, 5200],
        'July_Week3': [42500, 31500, 500, None, 123, 510, 0.489, 100, 84800000, 137000, 137000, 5200],
        'July_Week4': [42500, 31500, 0, 10800, 148, 515, 0.51, 100, 82400000, None, 135000, 5200]
    }
    return pd.DataFrame(data)

# Function to perform ARIMA forecasting
def arima_forecast(metric_data):
    model = ARIMA(metric_data, order=(5,1,0))  # ARIMA(5,1,0) model
    model_fit = model.fit()
    forecast = model_fit.forecast(steps=4)  # Forecast for next 4 weeks
    return forecast

# Streamlit app
def main():
    st.title('ARIMA Forecasting Dashboard')

    # Load data
    df = load_data()

    # Select metrics for dashboard
    selected_metrics = st.multiselect('Select Metrics for Dashboard:', df['Metrics'], default=df['Metrics'][:3])

    # Plot original data for selected metrics
    for metric in selected_metrics:
        st.subheader(f'Original Data for {metric}')
        plt.figure(figsize=(10, 6))
        plt.plot(df.columns[1:], df[df['Metrics'] == metric].iloc[:, 1:].values.flatten(), label='Original Data')
        plt.xticks(rotation=45)
        plt.legend()
        st.pyplot()

    # Perform ARIMA forecasting for selected metrics
    for metric in selected_metrics:
        st.subheader(f'ARIMA Forecast for {metric}')
        forecast = arima_forecast(df[df['Metrics'] == metric].iloc[:, 1:].values.flatten())
        forecast_index = [f'Week {i+1}' for i in range(len(df.columns[1:]), len(df.columns[1:])+4)]
        forecast_df = pd.DataFrame({'Week': forecast_index, 'Forecast': forecast})
        st.write(forecast_df)

if __name__ == '__main__':
    main()
