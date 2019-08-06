# StockMarket.Generic ![GitHub release](https://img.shields.io/github/release/stevewhit/stockmarket.generic.svg?color=green&style=popout)
StockMarket.Generic is a C# .NET library that contains generic components to create stock market desktop or web applications. Some of these components include:
* IEXCloud.io Stock Market Company & Data Downloader
* EntityFramework 6 DataModel & SQL Server Database Creation Scripts

## Dependencies
There is one project dependency to consider when installing & running this project. Both projects should reside in the same root folder in order to build properly.

1. [Framework.Generic](https://github.com/stevewhit/Framework.Generic)

## Installation
1. Create a free account with <a href="https://iexcloud.io">IEXCloud.io</a> and get a free API token.

2. Clone the repository and all [dependencies](#dependencies) to a single root folder.

```bash
git clone https://github.com/stevewhit/Framework.Generic
git clone https://github.com/stevewhit/StockMarket.Generic
```

3. Install NPM packages
```bash
npm install
```

4. Update application config file with your API token.
```config 
** StockMarket.Generic -> App.config **

<!-- Remove -->
<connectionStrings configSource="secretConnectionStrings.config" />

<!-- Add -->
<connectionStrings>
  <appSettings>
    <add key="IEXCloudToken" value="[YOUR_API_TOKEN_HERE]" />
    <add key="IEXCloudTokenTest" value="[YOUR_TEST_API_TOKEN_HERE]"/>
  </appSettings>
<connectionStrings/>
```

5. Run the SQL Server database creation scripts found in StockMarket.DataModel."Database Scripts"."Creation Scripts" and all subdirectories to create the SQL Server Database.

6. Follow installation steps for [Framework.Generic](https://github.com/stevewhit/Framework.Generic)

## Attribution to IEX
Stock market and company data for this application is [Powered by IEX Cloud](https://iexcloud.io).

View the [IEXCloud.io API](https://iexcloud.io/docs/api/#introduction).

## Lincense
Distributed under the MIT License. See [MIT License](https://choosealicense.com/licenses/mit/) for more information.

