import pandas as pd
import geopandas as gpd
import folium
from folium.plugins import HeatMap
from geodatasets import get_path


#  Load & Preprocess Data
df = pd.read_csv(
    "C:/Users/asati/OneDrive/Documents/Python/Visualization/nyc-traffic-collision-vizualization/Motor_Vehicle_Collisions_-_Crashes_20251208.csv"
)

# Drop rows missing coordinates
df = df.dropna(subset=["LATITUDE", "LONGITUDE"])

#Full Severity Formula
df["severity"] = (
    df["NUMBER OF PERSONS INJURED"]
    + df["NUMBER OF PEDESTRIANS INJURED"]
    + df["NUMBER OF CYCLIST INJURED"]
    + df["NUMBER OF MOTORIST INJURED"]
    + 3 * (
        df["NUMBER OF PERSONS KILLED"]
        + df["NUMBER OF PEDESTRIANS KILLED"]
        + df["NUMBER OF CYCLIST KILLED"]
        + df["NUMBER OF MOTORIST KILLED"]
    )
)

# Only keep meaningful events
df = df[df["severity"] > 0]

# 2️ Hotspot Heatmap of Collisions
heatmap_points = df[["LATITUDE", "LONGITUDE", "severity"]].values.tolist()

nyc_center = [40.7128, -74.0060]  # Center of NYC
hotspot_map = folium.Map(location=nyc_center, zoom_start=11)

HeatMap(
    heatmap_points,
    min_opacity=0.3,
    radius=10,
    blur=15,
    max_zoom=12
).add_to(hotspot_map)

hotspot_map.save("collision_hotspots_map.html")
print("Generated: collision_hotspots_map.html ")


# 3️ Borough Severity Choropleth
# Load borough boundaries and fix name
boro = gpd.read_file(get_path("nybb"))
boro = boro.to_crs(epsg=4326)
boro = boro.rename(columns={"BoroName": "BOROUGH"})

# Group severity by borough
borough_stats = df.groupby("BOROUGH")["severity"].sum().reset_index()

# Merge into GeoDataFrame
boro = boro.merge(borough_stats, on="BOROUGH", how="left")

# Create map
severity_map = folium.Map(location=nyc_center, zoom_start=10)

# Choropleth map with legend
choropleth = folium.Choropleth(
    geo_data=boro.to_json(),
    data=borough_stats,
    columns=["BOROUGH", "severity"],
    key_on="feature.properties.BOROUGH",
    fill_color="YlOrRd",  # Yellow → Red (low → high severity)
    fill_opacity=0.8,
    line_opacity=0.5,
    legend_name="Collision Severity (Injuries + 3×Fatalities)"
).add_to(severity_map)

# Tooltip on hover
folium.GeoJson(
    data=boro,
    tooltip=folium.features.GeoJsonTooltip(
        fields=["BOROUGH", "severity"],
        aliases=["Borough:", "Severity Score:"],
        localize=True,
        sticky=True
    ),
    name="Details"
).add_to(severity_map)

# Toggle layers
folium.LayerControl().add_to(severity_map)

severity_map.save("borough_severity_map.html")
print("Generated: borough_severity_map.html ")
