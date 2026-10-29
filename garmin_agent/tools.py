import os
import logging
from datetime import date, datetime
from typing import Optional, Dict, Any, List
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Import Garmin Connect library
from garminconnect import (
    Garmin,
    GarminConnectAuthenticationError,
    GarminConnectConnectionError,
    GarminConnectTooManyRequestsError,
    GarminConnectInvalidFileFormatError,
)

logger = logging.getLogger("garmin_agent_tools")

# Global Garmin API instance
garmin_api: Optional[Garmin] = None
is_authenticated = False

def garmin_login(random_string: str = "dummy") -> Dict[str, Any]:
    """Authenticate with Garmin Connect using environment variables.
    
    This tool MUST be called first before any other tool to establish a connection.
    
    Args:
        random_string (str, optional): A dummy parameter for consistency. Defaults to "dummy".
        
    Returns:
        dict: A dictionary containing the status and a message.
    """
    global garmin_api, is_authenticated
    
    print("\n🔑 [Tool] Authenticating with Garmin Connect...")
    
    email = os.getenv("GARMIN_EMAIL")
    password = os.getenv("GARMIN_PASSWORD")
    
    if not email or not password:
        print("❌ [Tool] Missing GARMIN_EMAIL or GARMIN_PASSWORD environment variables")
        return {
            "status": "error",
            "message": "Missing GARMIN_EMAIL or GARMIN_PASSWORD environment variables"
        }
    
    try:
        # Use garth directly for authentication
        from garth import Client as GarthClient
        garth_client = GarthClient()
        garth_client.login(email, password)
        
        # Now create Garmin instance with authenticated garth client
        garmin_api = Garmin(email, password)
        garmin_api.garth = garth_client  # Use the authenticated garth client
        is_authenticated = True
        
        print("✅ [Tool] Successfully authenticated with Garmin Connect!")
        return {
            "status": "success",
            "message": "Successfully authenticated with Garmin Connect"
        }
        
    except Exception as e:
        is_authenticated = False
        print(f"❌ [Tool] Authentication failed: {str(e)}")
        return {
            "status": "error", 
            "message": f"Authentication failed: {str(e)}"
        }

def _check_authentication():
    """Check if user is authenticated, raise exception if not"""
    if not is_authenticated or not garmin_api:
        # Try to auto-login if credentials are in env
        login_result = garmin_login()
        if login_result["status"] != "success":
            raise Exception("Not authenticated. Please call garmin_login first.")

def get_user_summary(date_str: str = "") -> Dict[str, Any]:
    """Get user activity summary for a specific date.
    
    Args:
        date_str (str, optional): Date in YYYY-MM-DD format. Defaults to today's date.
        
    Returns:
        dict: Daily activity summary including steps, calories, distance, etc.
    """
    try:
        _check_authentication()
        target_date = date_str if date_str else date.today().isoformat()
        print(f"📊 [Tool] Fetching daily activity summary for {target_date}...")
        result = garmin_api.get_user_summary(target_date)
        print("✅ [Tool] Daily activity summary fetched successfully!")
        return result
    except Exception as e:
        print(f"❌ [Tool] Failed to get user summary: {str(e)}")
        return {
            "status": "error",
            "message": f"Failed to get user summary: {str(e)}"
        }

def get_user_profile() -> Dict[str, Any]:
    """Get user profile and settings.
    
    Returns:
        dict: User profile details, settings, and physical metrics.
    """
    try:
        _check_authentication()
        print("👤 [Tool] Fetching user profile and settings...")
        result = garmin_api.get_user_profile()
        print("✅ [Tool] User profile fetched successfully!")
        return result
    except Exception as e:
        print(f"❌ [Tool] Failed to get user profile: {str(e)}")
        return {
            "status": "error",
            "message": f"Failed to get user profile: {str(e)}"
        }

def get_steps_data(date_str: str = "") -> Dict[str, Any]:
    """Get steps data for a specific date.
    
    Args:
        date_str (str, optional): Date in YYYY-MM-DD format. Defaults to today's date.
        
    Returns:
        dict: Detailed steps data for the specified date.
    """
    try:
        _check_authentication()
        target_date = date_str if date_str else date.today().isoformat()
        print(f"🚶‍♂️ [Tool] Fetching steps data for {target_date}...")
        result = garmin_api.get_steps_data(target_date)
        print("✅ [Tool] Steps data fetched successfully!")
        return result
    except Exception as e:
        print(f"❌ [Tool] Failed to get steps data: {str(e)}")
        return {
            "status": "error",
            "message": f"Failed to get steps data: {str(e)}"
        }

def get_heart_rate_data(date_str: str = "") -> Dict[str, Any]:
    """Get heart rate data for a specific date.
    
    Args:
        date_str (str, optional): Date in YYYY-MM-DD format. Defaults to today's date.
        
    Returns:
        dict: Detailed heart rate data for the specified date.
    """
    try:
        _check_authentication()
        target_date = date_str if date_str else date.today().isoformat()
        print(f"💓 [Tool] Fetching heart rate data for {target_date}...")
        result = garmin_api.get_heart_rates(target_date)
        print("✅ [Tool] Heart rate data fetched successfully!")
        return result
    except Exception as e:
        print(f"❌ [Tool] Failed to get heart rate data: {str(e)}")
        return {
            "status": "error",
            "message": f"Failed to get heart rate data: {str(e)}"
        }

def get_sleep_data(date_str: str = "") -> Dict[str, Any]:
    """Get sleep data for a specific date.
    
    Args:
        date_str (str, optional): Date in YYYY-MM-DD format. Defaults to today's date.
        
    Returns:
        dict: Detailed sleep data including sleep stages, duration, and score.
    """
    try:
        _check_authentication()
        target_date = date_str if date_str else date.today().isoformat()
        print(f"😴 [Tool] Fetching sleep data for {target_date}...")
        result = garmin_api.get_sleep_data(target_date)
        print("✅ [Tool] Sleep data fetched successfully!")
        return result
    except Exception as e:
        print(f"❌ [Tool] Failed to get sleep data: {str(e)}")
        return {
            "status": "error",
            "message": f"Failed to get sleep data: {str(e)}"
        }

def get_stress_data(date_str: str = "") -> Dict[str, Any]:
    """Get stress data for a specific date.
    
    Args:
        date_str (str, optional): Date in YYYY-MM-DD format. Defaults to today's date.
        
    Returns:
        dict: Detailed stress data for the specified date.
    """
    try:
        _check_authentication()
        target_date = date_str if date_str else date.today().isoformat()
        print(f"🧘‍♂️ [Tool] Fetching stress data for {target_date}...")
        result = garmin_api.get_stress_data(target_date)
        print("✅ [Tool] Stress data fetched successfully!")
        return result
    except Exception as e:
        print(f"❌ [Tool] Failed to get stress data: {str(e)}")
        return {
            "status": "error",
            "message": f"Failed to get stress data: {str(e)}"
        }

def get_body_battery(start_date: str, end_date: str = "") -> Dict[str, Any]:
    """Get body battery data for a date range.
    
    Args:
        start_date (str): Start date in YYYY-MM-DD format.
        end_date (str, optional): End date in YYYY-MM-DD format. Defaults to start_date.
        
    Returns:
        dict: Body battery metrics and charge/drain values.
    """
    try:
        _check_authentication()
        print(f"🔋 [Tool] Fetching body battery data from {start_date} to {end_date or start_date}...")
        if end_date:
            result = garmin_api.get_body_battery(start_date, end_date)
        else:
            result = garmin_api.get_body_battery(start_date)
        print("✅ [Tool] Body battery data fetched successfully!")
        return result
    except Exception as e:
        print(f"❌ [Tool] Failed to get body battery data: {str(e)}")
        return {
            "status": "error",
            "message": f"Failed to get body battery data: {str(e)}"
        }

def get_activities(start: int = 0, limit: int = 20, activity_type: str = "") -> Dict[str, Any]:
    """Get list of activities with optional filtering.
    
    Args:
        start (int, optional): Starting offset. Defaults to 0.
        limit (int, optional): Number of activities to return. Defaults to 20.
        activity_type (str, optional): Filter by activity type (cycling, running, swimming, etc.).
        
    Returns:
        dict: A dictionary containing the status and the list of activities or an error message.
    """
    try:
        _check_authentication()
        activity_filter = activity_type if activity_type else None
        print(f"🏃‍♂️ [Tool] Fetching recent activities (start={start}, limit={limit}, type={activity_filter or 'all'})...")
        activities = garmin_api.get_activities(start, limit, activity_filter)
        print(f"✅ [Tool] Successfully fetched {len(activities)} activities!")
        return {
            "status": "success",
            "activities": activities
        }
    except Exception as e:
        print(f"❌ [Tool] Failed to get activities: {str(e)}")
        return {
            "status": "error",
            "message": f"Failed to get activities: {str(e)}. Note: If you used an activity_type filter like 'mountain biking', it might be invalid. Try calling get_activities without a filter or with a broader filter like 'cycling'."
        }

def get_activity_details(activity_id: str) -> Dict[str, Any]:
    """Get detailed information about a specific activity.
    
    Args:
        activity_id (str): The unique activity ID.
        
    Returns:
        dict: Detailed activity metrics, splits, and GPS data.
    """
    try:
        _check_authentication()
        print(f"🔍 [Tool] Fetching detailed metrics for activity ID {activity_id}...")
        result = garmin_api.get_activity_details(activity_id)
        print("✅ [Tool] Activity details fetched successfully!")
        return result
    except Exception as e:
        print(f"❌ [Tool] Failed to get activity details: {str(e)}")
        return {
            "status": "error",
            "message": f"Failed to get activity details: {str(e)}"
        }

def get_activities_by_date(start_date: str, end_date: str = "", activity_type: str = "") -> Dict[str, Any]:
    """Get activities for a specific date range.
    
    Args:
        start_date (str): Start date in YYYY-MM-DD format.
        end_date (str, optional): End date in YYYY-MM-DD format. Defaults to start_date.
        activity_type (str, optional): Filter by activity type.
        
    Returns:
        dict: A dictionary containing the status and the list of activities or an error message.
    """
    try:
        _check_authentication()
        end_date_param = end_date if end_date else None
        activity_type_param = activity_type if activity_type else None
        print(f"📅 [Tool] Fetching activities from {start_date} to {end_date or start_date} (type={activity_type_param or 'all'})...")
        activities = garmin_api.get_activities_by_date(start_date, end_date_param, activity_type_param)
        print(f"✅ [Tool] Successfully fetched {len(activities)} activities within date range!")
        return {
            "status": "success",
            "activities": activities
        }
    except Exception as e:
        print(f"❌ [Tool] Failed to get activities by date: {str(e)}")
        return {
            "status": "error",
            "message": f"Failed to get activities by date: {str(e)}. Note: If you used an activity_type filter like 'mountain biking', it might be invalid. Try calling get_activities_by_date without a filter or with a broader filter like 'cycling'."
        }

def get_devices() -> Dict[str, Any]:
    """Get list of registered Garmin devices.
    
    Returns:
        dict: A dictionary containing the status and the list of devices or an error message.
    """
    try:
        _check_authentication()
        print("⌚ [Tool] Fetching registered Garmin devices...")
        devices = garmin_api.get_devices()
        print(f"✅ [Tool] Successfully fetched {len(devices)} devices!")
        return {
            "status": "success",
            "devices": devices
        }
    except Exception as e:
        print(f"❌ [Tool] Failed to get devices: {str(e)}")
        return {
            "status": "error",
            "message": f"Failed to get devices: {str(e)}"
        }

def add_weigh_in(weight: float, unit: str = "kg", timestamp: str = "") -> Dict[str, Any]:
    """Add a weight measurement.
    
    Args:
        weight (float): Weight value.
        unit (str, optional): Weight unit (kg, lbs). Defaults to "kg".
        timestamp (str, optional): Timestamp in ISO format. Defaults to now.
        
    Returns:
        dict: Status and result of the weigh-in operation.
    """
    try:
        _check_authentication()
        print(f"⚖️ [Tool] Adding weight measurement: {weight} {unit}...")
        
        if timestamp and timestamp.strip():
            ts = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
        else:
            ts = datetime.now()
        
        result = garmin_api.add_weigh_in(weight, unit, ts.isoformat())
        
        if hasattr(result, 'status_code'):
            response_data = {
                "status_code": result.status_code,
                "success": result.status_code in [200, 201, 204]
            }
        else:
            response_data = str(result)
            
        print("✅ [Tool] Weight measurement added successfully!")
        return {
            "status": "success",
            "message": "Weight measurement added successfully",
            "result": response_data
        }
    except Exception as e:
        print(f"❌ [Tool] Failed to add weight measurement: {str(e)}")
        return {
            "status": "error",
            "message": f"Failed to add weight measurement: {str(e)}"
        }

def download_activity(activity_id: str, format: str = "GPX") -> Dict[str, Any]:
    """Download an activity in a requested format (GPX, TCX, CSV, KML, ORIGINAL) and save it locally.
    
    Args:
        activity_id (str): The unique activity ID.
        format (str, optional): The download format. Supported values: "GPX", "TCX", "CSV", "KML", "ORIGINAL". Defaults to "GPX".
        
    Returns:
        dict: Status, local file path, and file size.
    """
    try:
        _check_authentication()
        
        fmt_upper = format.upper()
        if fmt_upper == "GPX":
            dl_fmt = Garmin.ActivityDownloadFormat.GPX
            ext = "gpx"
        elif fmt_upper == "TCX":
            dl_fmt = Garmin.ActivityDownloadFormat.TCX
            ext = "tcx"
        elif fmt_upper == "CSV":
            dl_fmt = Garmin.ActivityDownloadFormat.CSV
            ext = "csv"
        elif fmt_upper == "KML":
            dl_fmt = Garmin.ActivityDownloadFormat.KML
            ext = "kml"
        elif fmt_upper == "ORIGINAL":
            dl_fmt = Garmin.ActivityDownloadFormat.ORIGINAL
            ext = "zip"
        else:
            return {
                "status": "error",
                "message": f"Unsupported format: {format}. Supported formats are GPX, TCX, CSV, KML, ORIGINAL."
            }
            
        print(f"📥 [Tool] Downloading activity {activity_id} in {fmt_upper} format...")
        data = garmin_api.download_activity(activity_id, dl_fmt=dl_fmt)
        
        # Save file locally
        filename = f"activity_{activity_id}.{ext}"
        filepath = os.path.join(os.getcwd(), filename)
        
        with open(filepath, "wb") as f:
            f.write(data)
            
        print(f"✅ [Tool] Activity downloaded and saved to {filename} ({len(data)} bytes)!")
        return {
            "status": "success",
            "message": f"Activity downloaded successfully",
            "file_path": filepath,
            "file_name": filename,
            "file_size_bytes": len(data)
        }
    except Exception as e:
        print(f"❌ [Tool] Failed to download activity: {str(e)}")
        return {
            "status": "error",
            "message": f"Failed to download activity: {str(e)}"
        }
