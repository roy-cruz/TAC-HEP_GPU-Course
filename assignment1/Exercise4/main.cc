#include <stdio.h>
#include <iostream>
#include <vector>
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <stdexcept> //
#include <typeinfo> //

#include "t1.h"

#include <TMath.h>
#include <TFile.h>
#include <TTree.h>
#include <TH1F.h>
#include <TCanvas.h> 
#include <TLorentzVector.h>



//------------------------------------------------------------------------------
// Particle Class
//
class Particle{
	public:
		Particle();
		// FIXME : Create an additional constructor that takes 4 arguments --> the 4-momentum
		Particle(double px, double py, double pz, double energy);

		// Added factory since data provided is in pt, eta, phi, E format
		static Particle FromPtEtaPhiE(double pt, double eta, double phi, double energy);
		
		// Member variables
		double   pt, eta, phi, E, m, p[4]; 

		// Methods
		void     p4(double pt, double eta, double phi, double energy);
		void     print();
		void     setMass(double mass);
		double   sintheta();
};

class Lepton : public Particle {
	public: 
		Lepton() : Particle(), Q(0) {};
		Lepton(double px, double py, double pz, double energy, int charge) : Particle(px, py, pz, energy), Q(charge) {};
		static Lepton FromPtEtaPhiE(double pt, double eta, double phi, double energy, int charge);

		int Q;	
		void	setCharge(int charge);
};

class Jet : public Particle {
	public: 
		Jet() : Particle(), hadronFlav(-1) {};
		Jet(double px, double py, double pz, double energy, int hadronFlavor) : Particle(px, py, pz, energy), hadronFlav(hadronFlavor) {};
 		static Jet FromPtEtaPhiE(double pt, double eta, double phi, double energy, int hadronFlavor);

		int hadronFlav;
		void	setHadronFlavor(int flavor);
};


//------------------------------------------------------------------------------

//*****************************************************************************
//                                                                             *
//    MEMBERS functions of the Particle Class                                  *
//                                                                             *
//*****************************************************************************

//
//*** Default constructor ------------------------------------------------------
//
Particle::Particle(){
	pt = eta = phi = E = m = 0.0;
	p[0] = p[1] = p[2] = p[3] = 0.0;
}

//*** Additional constructor ------------------------------------------------------
Particle::Particle(double px, double py, double pz, double energy){ 
	
	if (energy < 1e-9){
		pt = eta = phi = E = m = 0.0;
		p[0] = p[1] = p[2] = p[3] = 0.0;
		return;
	}

	double m2 = energy*energy - (px*px + py*py + pz*pz);
	m = std::sqrt(m2);

	p[0] = energy;
	p[1] = px;
	p[2] = py;
	p[3] = pz;

	// Computing rest of quantities
	E = energy;
	pt = std::sqrt(px*px + py*pz);
	phi = std::atan2(py,px);
	eta = 0.5 * std::log((energy + pz)/(energy - pz));

}

Particle Particle::FromPtEtaPhiE(double pT, double eta, double phi, double E) {
	Particle obj;		   // default constructed
	obj.p4(pT, eta, phi, E);
	return obj;
}

//
//*** Members  ------------------------------------------------------
//
double Particle::sintheta() {
	double p_magnitude = std::sqrt(p[0]*p[0] + p[1]*p[1] + p[2]*p[2]);
	double sin_theta = (p_magnitude <1e-9) ? 0.0 : pt/p_magnitude;
	return sin_theta;
}

void Particle::p4(double pT, double eta, double phi, double energy){

	if (energy < 1e-9){
		pt = eta = phi = E = m = 0.0;
		p[0] = p[1] = p[2] = p[3] = 0.0;
		return;
	}

	double m2 = energy*energy - pT*pT * std::pow(std::cosh(eta),2);
	this->m = std::sqrt(m2);
	
	double px = pT * std::cos(phi);
	double py = pT * std::sin(phi);
	double pz = pT * std::sinh(eta);

	this->E = energy;
	this->pt = pT;
	this->eta = eta;
	this->phi = std::fmod(phi, M_PI);

	p[0] = energy;
	p[1] = px;
	p[2] = py;
	p[3] = pz;
}

void Particle::setMass(double mass)
{
	if (mass < 0) throw std::invalid_argument("setMass: negative mass");
	m = mass;
}

//
//*** Prints 4-vector ----------------------------------------------------------
//
void Particle::print(){
	std::cout << std::endl;
	std::cout << "(" << p[0] <<",\t" << p[1] <<",\t"<< p[2] <<",\t"<< p[3] << ")" << "  " <<  sintheta() << std::endl;
}

Lepton Lepton::FromPtEtaPhiE(double pt, double eta, double phi, double energy, int charge) {
	Lepton obj;
	obj.p4(pt, eta, phi, energy);
	obj.setCharge(charge);
	return obj;
}

void Lepton::setCharge(int charge) {
	Q = charge;
}

Jet Jet::FromPtEtaPhiE(double pt, double eta, double phi, double energy, int hadronFlavor) {
	Jet obj; // Runs Particle constructor
	obj.p4(pt, eta, phi, energy);
	obj.setHadronFlavor(hadronFlavor);
	return obj;
}

void Jet::setHadronFlavor(int hadronFlavor) {
	hadronFlav = hadronFlavor;
}

int main() {
	
	/* ************* */
	/* Input Tree   */
	/* ************* */

	std::cout << "Program start!" << std::endl;

	TFile *f      = new TFile("input.root","READ");
	TTree *t1 = (TTree*)(f->Get("t1"));

	// Read the variables from the ROOT tree branches
	t1->SetBranchAddress("lepPt",&lepPt);
	t1->SetBranchAddress("lepEta",&lepEta);
	t1->SetBranchAddress("lepPhi",&lepPhi);
	t1->SetBranchAddress("lepE",&lepE);
	t1->SetBranchAddress("lepQ",&lepQ);
	
	t1->SetBranchAddress("njets",&njets);
	t1->SetBranchAddress("jetPt",&jetPt);
	t1->SetBranchAddress("jetEta",&jetEta);
	t1->SetBranchAddress("jetPhi",&jetPhi);
	t1->SetBranchAddress("jetE", &jetE);
	t1->SetBranchAddress("jetHadronFlavour",&jetHadronFlavour);

	// Total number of events in ROOT tree
	Long64_t nentries = t1->GetEntries();
	std::cout << "Number of entries: " << nentries << std::endl;

	for (Long64_t jentry=0; jentry<(Long64_t)nentries/100; jentry++)
 	{
		t1->GetEntry(jentry); // Changes the address in address in memory each lep and jet var is refering to
		std::cout<<"Event "<< jentry <<std::endl;	

		//FIX ME
		std::cout << "   Printing lepton kinematics: " << std::endl;
		for (Long64_t ilep=0; ilep<maxLepSize; ilep++) {
			Lepton lepton = Lepton::FromPtEtaPhiE(lepPt[ilep], lepEta[ilep], lepPhi[ilep], lepE[ilep], lepQ[ilep]);
			if (lepton.E == 0.0) continue; // skip uninitialized leptons
			lepton.print();
			std::cout << "   Charge: " << lepQ[ilep] << std::endl;
		}

		std::cout << std::endl;

		std::cout << "   Printing jet kinematics: " << std::endl;
		for (Long64_t ijet=0; ijet<maxJetSize; ijet++) {
			Particle jet = Jet::FromPtEtaPhiE(jetPt[ijet], jetEta[ijet], jetPhi[ijet], jetE[ijet], jetHadronFlavour[ijet]);
			if (jet.E == 0.0) continue; // skip uninitialized jets
			jet.print();
			std::cout << "   Hadron Flavor: " << jetHadronFlavour[ijet] << std::endl;
		}
		std::cout << "--------------------------------------------------" << std::endl;
	} // Loop over all events

  	return 0;
}
