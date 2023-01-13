import streamlit as st
from pulldata import YahooFinanceHistory
from getmodel import Model

class APP():
	def __init__(self, name="Stock Price Prediction"):
		self.title = name

	def build(self):

		# APP name
		st.title(self.title)

		# Create Sidebar
		with st.sidebar:
			st.header('Input Parameters')
			stock = st.text_input('Stock code', 'AAPL')
			time = st.slider('Days back', 0, 1800, 30)
			lag = st.slider('Prediction lag', 1,5,1)
			epochs = st.slider('Training epochs', 10,100,10)
			optimizer = st.selectbox('Plearse select optimizer:',('adam','RMSprop'))
			features = st.multiselect('Please select training features:',['Open', 'High', 'Low', 'Close', 'Adj Close','Volume'])
			target = st.selectbox('Target to predict:',('Open','High','Low','Close','Adj Close','Volume'))
			start = st.button('Start Prediction')


		# Create Expander
		with st.expander("See Historical Data"):

			df = YahooFinanceHistory('AAPl',int(time)).get_quote()

			st.write(df)

		# Wait for training results
		st.header('Training Resutls:')

		if start == True:
			with st.spinner('Training in progress...'):
				model = Model(df, lag = lag, features = features, target = target, epochs = epochs, optimizer=optimizer)
				# model.checkdata()
				model.splitdata()
				model.processdata()
				model.train()
				fig = model.evaluate()

				st.write(fig)
			st.success('Done!')

			st.header('Prediction for tomorrow: ')
			price, today = model.tomorrow()
			st.metric(label= f'{target}', value= str(price), delta = str(round(price - today,2)))

if __name__ == '__main__':
	app = APP()
	app.build()