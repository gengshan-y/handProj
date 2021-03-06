/******************************************************************************
 * Author: Martin Godec
 *         godec@icg.tugraz.at
 ******************************************************************************/

#include "parameters.hpp"

Parameters::Parameters(const std::string& confFile) : m_filename(confFile)
{
	try{
		m_configFile.readFile(m_filename.c_str());
   } catch (libconfig::ParseException& e) {
        std::cout << "Error reading Configuration file (" << m_filename << ")!" << std::endl;
        std::cout << "ParseException at Line " << e.getLine() << ": " << e.getError() << std::endl;
        exit(EXIT_FAILURE);
    } catch (libconfig::SettingException& e) {
        std::cout << "Error reading Configuration file (" << m_filename << ")!" << std::endl;
        std::cout << "SettingException at " << e.getPath() << std::endl;
        exit(EXIT_FAILURE);
    } catch (std::exception& e) {
        std::cout << "Error reading Configuration file (" << m_filename << ")!" << std::endl;
        std::cout << e.what() << std::endl;
        exit(EXIT_FAILURE);
    } catch (...) {
        std::cout << "Error reading Configuration file (" << m_filename << ")!" << std::endl;
        std::cout << "Unknown Exception" << std::endl;
        exit(EXIT_FAILURE);
	}
}

Parameters::~Parameters()
{
}

int Parameters::readIntParameter(std::string param_name) const
{
    try{
        return (int)m_configFile.lookup(param_name);
    } catch (libconfig::ParseException& e) {
        std::cout << "Error reading Configuration file (" << m_filename << ")!" << std::endl;
        std::cout << "ParseException at Line " << e.getLine() << ": " << e.getError() << std::endl;
        exit(EXIT_FAILURE);
    } catch (libconfig::SettingException& e) {
        std::cout << "Error reading Configuration file (" << m_filename << ")!" << std::endl;
        std::cout << "SettingException at " << e.getPath() << std::endl;
        exit(EXIT_FAILURE);
    } catch (std::exception& e) {
        std::cout << "Error reading Configuration file (" << m_filename << ")!" << std::endl;
        std::cout << e.what() << std::endl;
        exit(EXIT_FAILURE);
    } catch (...) {
        std::cout << "Error reading Configuration file (" << m_filename << ")!" << std::endl;
        std::cout << "Unknown Exception" << std::endl;
        exit(EXIT_FAILURE);
	}
}

double Parameters::readDoubleParameter(std::string param_name) const
{
    try{
        return (double)m_configFile.lookup(param_name);
    } catch (libconfig::ParseException& e) {
        std::cout << "Error reading Configuration file (" << m_filename << ")!" << std::endl;
        std::cout << "ParseException at Line " << e.getLine() << ": " << e.getError() << std::endl;
        exit(EXIT_FAILURE);
    } catch (libconfig::SettingException& e) {
        std::cout << "Error reading Configuration file (" << m_filename << ")!" << std::endl;
        std::cout << "SettingException at " << e.getPath() << std::endl;
        exit(EXIT_FAILURE);
    } catch (std::exception& e) {
        std::cout << "Error reading Configuration file (" << m_filename << ")!" << std::endl;
        std::cout << e.what() << std::endl;
        exit(EXIT_FAILURE);
    } catch (...) {
        std::cout << "Error reading Configuration file (" << m_filename << ")!" << std::endl;
        std::cout << "Unknown Exception" << std::endl;
        exit(EXIT_FAILURE);
	}
}

std::string Parameters::readStringParameter(std::string param_name) const
{
    try{
        return std::string((const char*)m_configFile.lookup(param_name));
    } catch (libconfig::ParseException& e) {
        std::cout << "Error reading Configuration file (" << m_filename << ")!" << std::endl;
        std::cout << "ParseException at Line " << e.getLine() << ": " << e.getError() << std::endl;
        exit(EXIT_FAILURE);
    } catch (libconfig::SettingException& e) {
        std::cout << "Error reading Configuration file (" << m_filename << ")!" << std::endl;
        std::cout << "SettingException at " << e.getPath() << std::endl;
        exit(EXIT_FAILURE);
    } catch (std::exception& e) {
        std::cout << "Error reading Configuration file (" << m_filename << ")!" << std::endl;
        std::cout << e.what() << std::endl;
        exit(EXIT_FAILURE);
    } catch (...) {
        std::cout << "Error reading Configuration file (" << m_filename << ")!" << std::endl;
        std::cout << "Unknown Exception" << std::endl;
        exit(EXIT_FAILURE);
	}
}

bool Parameters::settingExists(std::string param_name) const
{
	return m_configFile.exists(param_name);
}

