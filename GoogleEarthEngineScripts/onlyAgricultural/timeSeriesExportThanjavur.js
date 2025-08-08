// References
// 1. https://custom-scripts.sentinel-hub.com/sentinel-2/
// 2. https://eo4society.esa.int/wp-content/uploads/2022/01/HYDR03_Drought_Monitoring_Tutorial.pdf


// Define the area of interest (Thanjavur district)
var thanjavur = ee.FeatureCollection('FAO/GAUL/2015/level2')
  .filter(ee.Filter.eq('ADM2_NAME', 'Thanjavur'))
  .filter(ee.Filter.eq('ADM1_NAME', 'Tamil Nadu')); // Adjust state name if needed

// Define years for extraction
var years = ['2015','2016', '2017', '2018', '2019', '2020', '2021', '2022', '2023', '2024', '2025'];

// Load a land cover dataset (e.g., Copernicus Global Land Cover)
var landCover = ee.Image('COPERNICUS/Landcover/100m/Proba-V-C3/Global/2019')
  .select('discrete_classification');

// Define agricultural land classes based on dataset documentation
var agricultureClasses = ee.List([40]); // Adjust based on your dataset's class values

// Create a mask for agricultural land
var agricultureMask = landCover.remap(agricultureClasses, ee.List.repeat(1, agricultureClasses.size()), 0);

// Clip the mask to the Thanjavur district
var agriculturalLand = agricultureMask.updateMask(agricultureMask).clip(thanjavur.geometry());

// Updated cloud masking function
function maskClouds(image) 
{
  // Check if the QA60 band is available
  var bandNames = image.bandNames();
  var hasQA60 = bandNames.contains('QA60');

  // Apply cloud masking if QA60 is available
  return ee.Image(ee.Algorithms.If(hasQA60,
    image.updateMask(image.select('QA60').not()),
    image  // Return the image unmodified if QA60 is not available
  ));
}

// Function to calculate indices
function calculateIndices(image) 
{
  //1. Normalized Difference Vegetation Index(-1 to 1)
  // https://custom-scripts.sentinel-hub.com/sentinel-2/ndvi/
  var ndvi = image.normalizedDifference(['B8', 'B4']).rename('NDVI');
  
  // 2. Enhanced Vegetation Index (-1 to 1)
  // https://custom-scripts.sentinel-hub.com/sentinel-2/evi/
  // Note: EVI used for dense area
  var evi = image.expression(
    '2.5 * (NIR - RED) / ((NIR + 6 * RED - 7.5 * BLUE) + 1)',
    {'NIR': image.select('B8'), 'RED': image.select('B4'), 'BLUE': image.select('B2')}
  ).rename('EVI');
  
  // 3. Atmospherically Resistant Vegetation Index (-1 to 1)
  // Reference : https://custom-scripts.sentinel-hub.com/sentinel-2/arvi/
  // Note : Used in area with high aerosol content
  var arvi = image.expression(
    '(NIR - RED - y * (RED - BLUE)) / (NIR + RED - y * (RED - BLUE))', 
    {
      'NIR': image.select('B8'),   // NIR band (Sentinel-2 band 8)
      'RED': image.select('B4'),   // Red band (Sentinel-2 band 4)
      'BLUE': image.select('B2'),  // Blue band (Sentinel-2 band 2)
      'y': 0.1  // Weight parameter (adjust according to your needs)
    }).rename('ARVI');
  
  // 4. Normalized Difference Water Index(-1 to 1)
  // Reference : https://custom-scripts.sentinel-hub.com/custom-scripts/sentinel-2/ndwi/
  // Note : Used for water areas. Have to check whether to use it or not
  var ndwi = image.normalizedDifference(['B3', 'B8']).rename('NDWI');
  
  // 5. Soil Adjusted Vegetation Index 
  // Reference : https://custom-scripts.sentinel-hub.com/sentinel-2/savi/
  // NIR : B8, Red : B4
  var savi = image.expression(
    '((NIR - RED) / (NIR + RED + l)) * (1 + l)',
    {'NIR': image.select('B8'), 'RED': image.select('B4'), 'l': 0.8}
  ).rename('SAVI');
  
  // 6. Transformed Vegetative Index
  // Reference : https://www.indexdatabase.de/db/si-single.php?sensor_id=96&rsindex_id=98
  var tvi = ndvi.expression('sqrt(NDVI + 0.5)', {
    'NDVI': ndvi
  }).rename('TVI');

  // 7. Normalized Difference Moisture Index
  // Reference : https://custom-scripts.sentinel-hub.com/sentinel-2/ndmi/
  // NIR : B8, SWIR1 : B11
  var ndmi = image.normalizedDifference(['B8', 'B11']).rename('NDMI');
  
  // 8. Normalized multi-band drought index 
  // Reference : https://www.researchgate.net/publication/248815330_NMDI_A_normalized_multi-band_drought_index_for_monitoring_soil_and_vegetation_moisture_with_satellite_remote_sensing
  // NIR : B8, SWIR1 : 11, SWIR2: 12
  var nmdi = image.expression(
    '(NIR - (SWIR1 - SWIR2)) / (NIR + (SWIR1 - SWIR2))',
    {'NIR': image.select('B8'), 'SWIR1': image.select('B11'), 'SWIR2': image.select('B12')}
  ).rename('NMDI');

  // 9. Modified Normalized Water Index 
  // Reference : https://eo4society.esa.int/wp-content/uploads/2022/01/HYDR03_Drought_Monitoring_Tutorial.pdf
  // Green : B3, SWIR1 : B11
  var mndwi = image.normalizedDifference(['B3', 'B11']).rename('MNDWI');
  
  // 10. Normalized Difference NIR/MIR Modified Normalized Difference Vegetation Index
  // https://www.indexdatabase.de/db/si-single.php?sensor_id=96&rsindex_id=245
  // NIR : B9, MIR: B12
  var mndvi = image.normalizedDifference(['B9', 'B12']).rename('MNDVI');
  
  // 11. Ratio Drought Index 
  // Reference : https://www.indexdatabase.de/db/si-single.php?sensor_id=96&rsindex_id=71
  var rdi = image.expression(
    '(MIR/NIR)',
    {'NIR': image.select('B9'), 'MIR': image.select('B12')}
  ).rename('RDI');

  // 12.  Red-edge Chlorophyll Index 
  // Reference : https://custom-scripts.sentinel-hub.com/custom-scripts/sentinel-2/chl_rededge/
  var reci = image.expression(
    '((NIR / RedEdge) - 1)', {
      'NIR': image.select('B7'),
      'RedEdge': image.select('B5')
    }
  ).rename('RECI');

  // Add all indices as bands
  var allIndices = image.addBands([arvi, evi, mndvi, mndwi, ndmi, ndvi, ndwi, nmdi, rdi, reci, savi, tvi]);

  // Clip only once to Thanjavur district
  //return allIndices.clip(thanjavur.geometry());
  return allIndices.updateMask(agriculturalLand).clip(thanjavur.geometry());
}

// Function to filter and process the image collection for each year
function getYearlyCollection(year, region) {
  var startDate = year + '-01-01';
  var endDate = year + '-12-30';
  
  return ee.ImageCollection('COPERNICUS/S2')
    .filterBounds(region)
    .filterDate(startDate, endDate)
    .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 20))
    .map(maskClouds)  // Apply cloud masking
    .map(calculateIndices);
}

// Function to create a time series for each index and year
function extractTimeSeries(collection, indices, region, year) 
{
  var timeSeries = collection.map(function(image) 
  {
    // Extract date from image
    var date = ee.Date(image.get('system:time_start')).format('YYYY-MM-dd');
    // Reduce the image to the region mean
    var stats = image.reduceRegion({
      reducer: ee.Reducer.mean(),
      geometry: region,
      scale: 10,
      bestEffort: true
    });
    // Create a feature with the date and indices
    var feature = ee.Feature(null, {'date': date});
    indices.forEach(function(index) 
    {
      feature = feature.set(index, stats.get(index));
    });
    return feature;
  });

  return timeSeries;
}

// Loop over each year and extract the data
years.forEach(function(year) {
  var collection = getYearlyCollection(year, thanjavur);
  print('Thanjavur Image Count:', getYearlyCollection(year, thanjavur).size());
  
  // Indices to extract
  var indices = ['ARVI','EVI','MNDVI','MNDWI','NDMI','NDVI','NDWI','NMDI','RDI','RECI','SAVI', 'TVI'];

  var timeSeries = extractTimeSeries(collection, indices, thanjavur, year);

  // Export to Google Drive
  Export.table.toDrive({
    collection: ee.FeatureCollection(timeSeries),
    description: 'TimeSeries_' + year,
    folder: 'L1C',
    fileNamePrefix: 'TimeSeries_Thanjavur_' + year,
    fileFormat: 'CSV'
  });
});
