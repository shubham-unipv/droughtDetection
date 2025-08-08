// Sentinel-2 3-day Mosaic Time Series Export for Thanjavur District

var thanjavur = ee.FeatureCollection('FAO/GAUL/2015/level2')
  .filter(ee.Filter.eq('ADM2_NAME', 'Thanjavur'))
  .filter(ee.Filter.eq('ADM1_NAME', 'Tamil Nadu'));

var years = ['2015','2016', '2017', '2018', '2019', '2020', '2021', '2022', '2023', '2024', '2025'];
var indices = ['ARVI','EVI','MNDVI','MNDWI','NDMI','NDVI','NDWI','NMDI','RDI','RECI','SAVI', 'TVI'];

// Land cover mask setup
var landCover = ee.Image('COPERNICUS/Landcover/100m/Proba-V-C3/Global/2019')
  .select('discrete_classification');
var agricultureClasses = ee.List([40]);
var agricultureMask = landCover.remap(agricultureClasses, ee.List.repeat(1, agricultureClasses.size()), 0);
var agriculturalLand = agricultureMask.updateMask(agricultureMask).clip(thanjavur.geometry());

function hasBands(image, bandNames) {
  // Handle null/undefined cases
  if (!image) return ee.Number(0);
  
  // Ensure we have valid inputs
  var imgBands = ee.List(image.bandNames());
  bandNames = ee.List(bandNames);
  
  // Check if all required bands exist
  var allExist = ee.Number(
    bandNames.map(function(b) {
      return ee.Number(imgBands.contains(b));
    }).reduce(ee.Reducer.min())
  );
  
  return allExist;
}

function maskClouds(image) {
  var bandNames = image.bandNames();
  var hasQA60 = bandNames.contains('QA60');
  
  // Add dummy QA60 band if missing
  var imageWithQA60 = ee.Image(ee.Algorithms.If(
    hasQA60,
    image,
    image.addBands(ee.Image(0).rename('QA60').updateMask(ee.Image(1)).uint16())
  ));
  
  // Apply cloud mask
  return imageWithQA60.updateMask(imageWithQA60.select('QA60').not());
}

function calculateIndices(image) {
  // First check if image exists
  if (!image) return null;
  
  var requiredBands = ['B2','B3','B4','B5','B7','B8','B9','B11','B12'];
  var hasAll = hasBands(image, requiredBands);
  
  return ee.Image(ee.Algorithms.If(hasAll,
    image.addBands([
      image.normalizedDifference(['B8', 'B4']).rename('NDVI'),
      image.expression(
        '2.5 * (NIR - RED) / ((NIR + 6 * RED - 7.5 * BLUE) + 1)',
        {
          'NIR': image.select('B8'),
          'RED': image.select('B4'),
          'BLUE': image.select('B2')
        }
      ).rename('EVI'),
      image.expression(
        '(NIR - RED - y * (RED - BLUE)) / (NIR + RED - y * (RED - BLUE))',
        {
          'NIR': image.select('B8'),
          'RED': image.select('B4'),
          'BLUE': image.select('B2'),
          'y': 0.1
        }
      ).rename('ARVI'),
      image.normalizedDifference(['B3', 'B8']).rename('NDWI'),
      image.expression(
        '((NIR - RED) / (NIR + RED + l)) * (1 + l)',
        {
          'NIR': image.select('B8'),
          'RED': image.select('B4'),
          'l': 0.8
        }
      ).rename('SAVI'),
      image.normalizedDifference(['B8', 'B4']).expression('sqrt(b(0) + 0.5)').rename('TVI'),
      image.normalizedDifference(['B8', 'B11']).rename('NDMI'),
      image.expression(
        '(NIR - (SWIR1 - SWIR2)) / (NIR + (SWIR1 - SWIR2))',
        {
          'NIR': image.select('B8'),
          'SWIR1': image.select('B11'),
          'SWIR2': image.select('B12')
        }
      ).rename('NMDI'),
      image.normalizedDifference(['B3', 'B11']).rename('MNDWI'),
      image.normalizedDifference(['B9', 'B12']).rename('MNDVI'),
      image.expression(
        'MIR/NIR',
        {
          'NIR': image.select('B9'),
          'MIR': image.select('B12')
        }
      ).rename('RDI'),
      image.expression(
        '(NIR / RedEdge) - 1',
        {
          'NIR': image.select('B7'),
          'RedEdge': image.select('B5')
        }
      ).rename('RECI')
    ]),
    null
  ));
}

function getCloudPercent(image) {
  if (!image) return ee.Number(0);
  
  var cloudMean = ee.Number(
    image.select(['QA60']).reduceRegion({
      reducer: ee.Reducer.mean(),
      geometry: thanjavur.geometry(),
      scale: 10,
      bestEffort: true,
      maxPixels: 1e9
    }).get('QA60')
  );
  
  return ee.Algorithms.If(
    cloudMean,
    ee.Number(cloudMean).multiply(100),
    ee.Number(0)
  );
}

var standardBands = ['B1','B2','B3','B4','B5','B6','B7','B8','B8A','B9','B10','B11','B12','QA60'];

// Process each year
years.forEach(function(year) {
  var start = ee.Date(year + '-01-01');
  var end = ee.Date(year + '-12-31');
  var days = end.difference(start, 'day');
  var nPeriods = days.divide(3).ceil();
  
  // Create time windows
  var windowsList = ee.List.sequence(0, nPeriods.subtract(1));
  var windows = windowsList.map(function(i) {
    var windowStart = start.advance(ee.Number(i).multiply(3), 'day');
    var windowEnd = windowStart.advance(3, 'day');
    return ee.Dictionary({
      'year': year,
      'windowStart': windowStart.format('YYYY-MM-dd'),
      'windowEnd': windowEnd.format('YYYY-MM-dd')
    });
  });

  // Process windows
  var features = ee.FeatureCollection(windows.map(function(dict) {
    dict = ee.Dictionary(dict);
    var windowStart = ee.Date(dict.getString('windowStart'));
    var windowEnd = ee.Date(dict.getString('windowEnd'));

    var collection = ee.ImageCollection('COPERNICUS/S2')
      .filterBounds(thanjavur)
      .filterDate(windowStart, windowEnd)
      .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 80))
      .map(function(img) {
        var imgBands = img.bandNames();
        var hasQA60 = imgBands.contains('QA60');
        img = ee.Image(ee.Algorithms.If(
          hasQA60,
          img,
          img.addBands(ee.Image(0).rename('QA60').updateMask(ee.Image(1)).uint16())
        ));
        return img.select(standardBands);
      })
      .map(maskClouds);

    return ee.Feature(null, ee.Algorithms.If(
      collection.size().gt(0),
      function() {
        var mosaicImage = collection.median()
          .updateMask(agriculturalLand)
          .clip(thanjavur.geometry());
        
        var indicesImage = calculateIndices(mosaicImage);
        if (!indicesImage) return null;
        
        var cloudPercent = getCloudPercent(mosaicImage);
        
        return ee.Algorithms.If(
          ee.Number(cloudPercent).gt(20),
          null,
          function() {
            var stats = indicesImage.reduceRegion({
              reducer: ee.Reducer.mean(),
              geometry: thanjavur.geometry(),
              scale: 10,
              bestEffort: true,
              maxPixels: 1e9
            });
            
            var props = {
              'year': year,
              'date': windowStart.format('YYYY-MM-dd'),
              'cloudPercent': cloudPercent
            };
            
            // Safely add indices
            indices.forEach(function(index) {
              props[index] = ee.Number(stats.get(index));
            });
            
            return props;
          }()
        );
      }(),
      null
    ));
  }));
  // Filter null features and export
  var validFeatures = features.filter(ee.Filter.notNull(['date']));
  
  Export.table.toDrive({
    collection: validFeatures,
    description: 'MosaicTimeSeries_Thanjavur_' + year,
    folder: 'L1C',
    fileNamePrefix: 'MosaicTimeSeries_Thanjavur_' + year,
    fileFormat: 'CSV'
  });
});